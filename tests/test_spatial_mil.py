"""Independent paired-model, learning, masking, and checkpoint checks on CPU.

Torch tests explicitly skip when the optional training dependency is absent.
The real timm check uses random initialization and never downloads weights.
"""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

TORCH_PRESENT = importlib.util.find_spec("torch") is not None
if TORCH_PRESENT:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from rsna.spatial_mil import (SpatialTargetMIL, paired_models, preprocess_windows,
                                 save_checkpoint, load_checkpoint)


if TORCH_PRESENT:
    class FakeBackbone(nn.Module):
        """Spatial features with an explicit pooling-then-normalization head."""
        num_features = 8

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, self.num_features, 3, padding=1)
            self.head_norm = nn.LayerNorm(self.num_features)

        def forward_features(self, x):
            return torch.tanh(self.conv(x))

        def forward_head(self, maps, pre_logits=True):
            assert pre_logits
            return self.head_norm(F.adaptive_avg_pool2d(maps, 1).flatten(1))


@unittest.skipUnless(TORCH_PRESENT, "Optional Torch training dependency is not installed")
class SpatialMILTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1907)
        self.config = {"resolution": 16, "head_dropout": .2,
                       "spatial_key_dim": 4, "gradient_checkpointing": False}
        self.control, self.spatial = paired_models(self.config, backbone=FakeBackbone())
        self.windows = torch.randint(0, 256, (2, 4, 3, 20, 20), dtype=torch.uint8)
        self.slots = torch.tensor([[0, 1, 2, 5], [0, 3, 5, 5]])
        self.mask = self.slots != 5
        self.positions = torch.tensor([[-.7, -.2, .4, 0], [-.4, .6, 0, 0]])

    def forward_model(self, model):
        return model(self.windows, self.slots, self.mask, self.positions)

    def test_all_initial_parameter_bytes_and_zero_query_logits_match(self):
        for name, tensor in self.control.state_dict().items():
            self.assertTrue(torch.equal(tensor, self.spatial.state_dict()[name]), name)
        self.control.eval()
        self.spatial.eval()
        torch.testing.assert_close(self.forward_model(self.control), self.forward_model(self.spatial),
                                   rtol=0, atol=0)

    def test_paired_training_dropout_and_augmentation_replay_same_rng(self):
        self.control.train()
        self.spatial.train()
        state = torch.get_rng_state()
        first = self.control(self.windows, self.slots, self.mask, self.positions, augment=True)
        torch.set_rng_state(state)
        second = self.spatial(self.windows, self.slots, self.mask, self.positions, augment=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_uniform_control_matches_original_vector_MIL_formula(self):
        self.control.eval()
        features = torch.randn(2, 4, self.control.feature_dim)
        h = self.control.norm(features.float())
        h = h + self.control.slot(self.slots) + self.control.position(self.positions.unsqueeze(-1))
        scores = self.control.att_out(torch.tanh(self.control.att_v(h)) * torch.sigmoid(self.control.att_u(h)))
        scores = scores.masked_fill(~self.mask.unsqueeze(-1), -1e4)
        attention = scores.float().softmax(1) * self.mask.unsqueeze(-1)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bkl,bkd->bld", attention, h)
        original = (pooled*self.control.classifier_weight.unsqueeze(0)).sum(-1)+self.control.classifier_bias
        readouts = features[:, :, None].expand(-1, -1, 12, -1)
        current = self.control.pool_features(readouts, self.slots, self.mask, self.positions)
        torch.testing.assert_close(current, original, rtol=1e-5, atol=1e-6)

    def test_zero_queries_have_finite_nonzero_gradient_and_learn_after_step(self):
        self.spatial.train()
        optimizer = torch.optim.AdamW(self.spatial.parameters(), lr=.01)
        logits = self.forward_model(self.spatial)
        labels = torch.randint(0, 2, logits.shape).float()
        F.binary_cross_entropy_with_logits(logits, labels).backward()
        gradient = self.spatial.spatial_queries.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        before = self.spatial.spatial_queries.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, self.spatial.spatial_queries.detach()))

    def test_padding_does_not_vote_and_findings_have_separate_readouts(self):
        self.spatial.eval()
        features = torch.randn(2, 4, 12, self.spatial.feature_dim)
        baseline = self.spatial.pool_features(features, self.slots, self.mask, self.positions)
        changed_padding = features.clone()
        changed_padding[~self.mask] = torch.randn_like(changed_padding[~self.mask]) * 1000
        torch.testing.assert_close(baseline, self.spatial.pool_features(changed_padding, self.slots, self.mask, self.positions),
                                   rtol=0, atol=0)
        changed_target = features.clone()
        changed_target[:, :, 4, :] += torch.linspace(-5, 5, self.spatial.feature_dim)
        after = self.spatial.pool_features(changed_target, self.slots, self.mask, self.positions)
        other_targets = [j for j in range(12) if j != 4]
        torch.testing.assert_close(baseline[:, other_targets], after[:, other_targets], rtol=0, atol=0)
        self.assertGreater(float((baseline[:, 4]-after[:, 4]).abs().sum()), 0)

    def test_missing_views_remain_finite_but_all_invalid_study_is_rejected(self):
        self.spatial.eval()
        self.assertTrue(torch.isfinite(self.forward_model(self.spatial)).all())
        bad = self.mask.clone()
        bad[1] = False
        with self.assertRaises(ValueError):
            self.spatial(self.windows, self.slots, bad, self.positions)

    def test_normalization_gain_is_study_shared_and_preserves_channel_order(self):
        identical = torch.full((2, 3, 3, 10, 10), 100, dtype=torch.uint8)
        config = {"resolution": 8, "mean": [0, 0, 0], "std": [1, 1, 1]}
        x = preprocess_windows(identical, config, augment=True)
        for study in x:
            torch.testing.assert_close(study, torch.full_like(study, float(study.flatten()[0])), rtol=0, atol=2e-7)
        channels = torch.zeros((1, 1, 3, 4, 4), dtype=torch.uint8)
        channels[:, :, 1] = 128
        channels[:, :, 2] = 255
        y = preprocess_windows(channels, config)
        self.assertEqual(float(y[0, 0, 0].mean()), 0)
        self.assertAlmostEqual(float(y[0, 0, 1].mean()), 128/255, places=6)
        self.assertEqual(float(y[0, 0, 2].mean()), 1)

    def test_strict_checkpoint_roundtrip_and_contract_mismatch_rejection(self):
        self.spatial.eval()
        contract = {"split_hash": "synthetic", "pixel_contract": "synthetic-cache-v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(self.spatial, path, contract=contract)
            restored = SpatialTargetMIL(self.config, True, backbone=FakeBackbone())
            load_checkpoint(restored, path, contract=contract)
            restored.eval()
            torch.testing.assert_close(self.forward_model(self.spatial), self.forward_model(restored), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                load_checkpoint(restored, path, contract={"split_hash": "changed"})
            wrong_arm = SpatialTargetMIL(self.config, False, backbone=FakeBackbone())
            with self.assertRaises(ValueError):
                load_checkpoint(wrong_arm, path, contract=contract)
            changed = copy.deepcopy(self.config)
            changed["resolution"] = 20
            wrong_config = SpatialTargetMIL(changed, True, backbone=FakeBackbone())
            with self.assertRaises(ValueError):
                load_checkpoint(wrong_config, path, contract=contract)

    @unittest.skipUnless(importlib.util.find_spec("timm") is not None, "Optional timm training dependency is not installed")
    def test_real_convnext_post_pool_normalization_and_zero_query_parity(self):
        import timm
        backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0, global_pool="avg")
        model = SpatialTargetMIL({"head_dropout": 0, "spatial_key_dim": 8}, True, backbone=backbone)
        model.eval()
        with torch.no_grad():
            for resolution in (64, 96):
                images = torch.randn(1, 3, resolution, resolution)
                maps = backbone.forward_features(images)
                original = backbone.forward_head(maps, pre_logits=True)
                spatial = model.pool_spatial_maps(maps)
                torch.testing.assert_close(spatial, original[:, None].expand_as(spatial), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
