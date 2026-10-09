"""Meaningful synthetic training, serialization, and input-contract checks."""
import json
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsna.combiner import METHODS, fit_combiner, predict_combiner, validate_probabilities


class CombinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1937)
        latent = rng.normal(size=(800, 12))
        cls.labels = (latent > 0).astype(np.float64)
        cls.probabilities = np.stack([
            expit(latent + rng.normal(scale=2.0, size=latent.shape)),
            expit(latent + rng.normal(scale=0.3, size=latent.shape)),
            expit(latent + rng.normal(scale=1.0, size=latent.shape)),
            expit(latent + rng.normal(scale=1.5, size=latent.shape)),
        ], axis=1)
        cls.confidence = np.ones_like(cls.labels)

    def test_all_original_models_learn_useful_complementary_signal_on_unseen_rows(self):
        split = 600
        incumbent = np.mean([roc_auc_score(self.labels[split:, j], self.probabilities[split:, 0, j]) for j in range(12)])
        for method in METHODS:
            with self.subTest(method=method):
                model = fit_combiner(self.probabilities[:split], self.labels[:split], self.confidence[:split], method)
                predicted = predict_combiner(model, self.probabilities[split:])
                auc = np.mean([roc_auc_score(self.labels[split:, j], predicted[:, j]) for j in range(12)])
                self.assertGreater(auc, incumbent + 0.05)
                self.assertTrue(np.isfinite(predicted).all())
                restored = json.loads(json.dumps(model))
                np.testing.assert_array_equal(predict_combiner(restored, self.probabilities[split:]), predicted)
                if method == "convex_probability":
                    for head in model["heads"]:
                        self.assertAlmostEqual(sum(head["weights"]), 1.0)
                        self.assertTrue(all(w >= 0 for w in head["weights"]))

    def test_invalid_component_inputs_are_rejected(self):
        for value in (-0.1, 1.1, np.nan, np.inf):
            with self.subTest(value=value):
                corrupted = self.probabilities.copy()
                corrupted[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    validate_probabilities(corrupted)
        with self.assertRaises(ValueError):
            validate_probabilities(self.probabilities[:, :3])

    def test_ambiguous_or_single_class_labels_do_not_fit_fake_heads(self):
        ambiguous = np.full_like(self.labels, 0.5)
        with self.assertRaises(ValueError):
            fit_combiner(self.probabilities, ambiguous, self.confidence, "convex_probability")
        with self.assertRaises(ValueError):
            fit_combiner(self.probabilities, np.ones_like(self.labels), self.confidence, "logit_same_class")

    def test_group_bootstrap_auc_implementation_matches_weighted_reference_with_ties(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "fit_oof_combiner.py"
        spec = importlib.util.spec_from_file_location("fit_oof_combiner", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = np.random.default_rng(614)
        labels = self.labels[:50].copy()
        labels[0, :] = 0.5  # Ambiguous labels must be excluded, not coerced to negatives.
        tied_scores = np.floor(self.probabilities[:50, 0, :] * 4) / 4
        counts = rng.integers(0, 4, size=50)
        actual = module.auc_from_cache(module.make_auc_cache(labels, tied_scores), counts)
        expected = []
        for j in range(12):
            eligible = np.abs(labels[:, j]-0.5) > 1e-6
            expected.append(roc_auc_score(labels[eligible, j] > 0.5, tied_scores[eligible, j],
                                          sample_weight=counts[eligible]))
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


if __name__ == "__main__":
    unittest.main()
