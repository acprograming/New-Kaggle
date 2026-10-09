"""Original per-finding spatial attention ablation over the public sampled-plane cache.

The uniform and spatial arms share an identical ConvNeXt encoder and target-wise
slot/position MIL head. Spatial queries start at zero. Attention-minus-uniform
pooling keeps the initial output exactly uniform while retaining query gradients.
The actual backbone's forward_head preserves its pooling-then-normalisation order.
"""
from __future__ import annotations
import copy
import math
import os
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']


def preprocess_windows(windows: torch.Tensor, config: dict[str, Any], augment: bool = False) -> torch.Tensor:
    """uint8 B×K×3×H×W -> ImageNet-normalised B×K×3×resolution×resolution.

    Cache planes are sampled positions, not adjacent native DICOM slices.
    Gain augmentation is shared across every window/channel of each study.
    """
    if windows.ndim != 5 or windows.shape[2] != 3 or windows.dtype != torch.uint8:
        raise ValueError('Expected uint8 B×K×3×H×W windows')
    b, k = windows.shape[:2]
    x = windows.float().div(255.)
    if augment:
        gain = torch.empty((b, 1, 1, 1, 1), device=x.device).uniform_(.9, 1.1)
        x = (x * gain).clamp(0, 1)
    res = int(config.get('resolution', 320))
    x = F.interpolate(x.flatten(0, 1), size=(res, res), mode='bilinear', align_corners=False)
    mean = x.new_tensor(config.get('mean', [.485, .456, .406])).view(1, 3, 1, 1)
    std = x.new_tensor(config.get('std', [.229, .224, .225])).view(1, 3, 1, 1)
    if (std <= 0).any():
        raise ValueError('Normalisation standard deviations must be positive')
    return ((x - mean) / std).view(b, k, 3, res, res)


class SpatialTargetMIL(nn.Module):
    """Twelve spatial readouts, then the same per-finding gated window MIL head.

    A supplied backbone must expose num_features, forward_features(NCHW), and
    forward_head(NCHW, pre_logits=True). This injection avoids external downloads
    in CPU tests. The production backbone is a timm ConvNeXt with global_pool='avg'.
    """
    def __init__(self, config: dict[str, Any], spatial: bool, *, backbone: nn.Module | None = None,
                 pretrained: bool = False):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.spatial = bool(spatial)
        if backbone is None:
            import timm
            backbone = timm.create_model(config.get('architecture', 'convnext_tiny.fb_in22k_ft_in1k'),
                                         pretrained=pretrained, num_classes=0, global_pool='avg', in_chans=3)
        self.backbone = backbone
        self.feature_dim = dim = int(backbone.num_features)
        self.norm = nn.LayerNorm(dim)
        self.slot = nn.Embedding(6, dim, padding_idx=5)
        self.position = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, dim))
        self.att_v = nn.Linear(dim, 256)
        self.att_u = nn.Linear(dim, 256)
        self.att_out = nn.Linear(256, len(TARGETS))
        self.classifier_weight = nn.Parameter(torch.empty(len(TARGETS), dim))
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        self.dropout = nn.Dropout(float(config.get('head_dropout', .20)))
        nn.init.normal_(self.slot.weight, std=.01)
        with torch.no_grad():
            self.slot.weight[5].zero_()
        nn.init.trunc_normal_(self.classifier_weight, std=.02)
        # Create the extra spatial parameters last: ordinary MIL initialisation is unchanged.
        key_dim = int(config.get('spatial_key_dim', 64))
        self.spatial_key = nn.Linear(dim, key_dim, bias=False)
        self.spatial_queries = nn.Parameter(torch.zeros(len(TARGETS), key_dim))
        if not spatial:
            self.spatial_key.requires_grad_(False)
            self.spatial_queries.requires_grad_(False)
        if config.get('gradient_checkpointing', False):
            if not hasattr(backbone, 'set_grad_checkpointing'):
                raise ValueError('Backbone does not support gradient checkpointing')
            backbone.set_grad_checkpointing(True)

    def pool_spatial_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """N×D×H×W -> N×12×D using the backbone's own post-pool head.

        No normalisation is moved ahead of spatial pooling. Subtracting the
        uniform attention makes zero-query initialisation exact without blocking
        the derivative of attention with respect to the query parameters.
        """
        if maps.ndim != 4 or maps.shape[1] != self.feature_dim:
            raise ValueError('Expected backbone spatial N×D×H×W map')
        n, d, height, width = maps.shape
        base = F.adaptive_avg_pool2d(maps, 1).flatten(1)
        pooled = base[:, None, :].expand(n, len(TARGETS), d)
        if self.spatial:
            tokens = maps.flatten(2).transpose(1, 2)
            keys = self.spatial_key(F.layer_norm(tokens.float(), (d,)).to(tokens.dtype))
            scores = torch.einsum('npd,cd->npc', keys, self.spatial_queries.to(keys.dtype)) / math.sqrt(keys.shape[-1])
            attention = scores.float().softmax(dim=1)
            delta = attention - (1. / (height * width))
            residual = torch.einsum('npc,npd->ncd', delta.to(tokens.dtype), tokens)
            pooled = pooled + residual
        head_input = pooled.reshape(n * len(TARGETS), d, 1, 1).contiguous()
        out = self.backbone.forward_head(head_input, pre_logits=True)
        if out.shape != (n * len(TARGETS), d):
            raise ValueError('Backbone forward_head changed its feature contract')
        return out.view(n, len(TARGETS), d)

    def encode_windows(self, normalized_images: torch.Tensor, chunk: int = 0) -> torch.Tensor:
        """Encode real windows only; never send padded windows through the encoder."""
        if len(normalized_images) == 0:
            raise ValueError('No valid windows to encode')
        step = chunk or len(normalized_images)
        if step < 1:
            raise ValueError('Chunk size must be positive')
        return torch.cat([self.pool_spatial_maps(self.backbone.forward_features(normalized_images[i:i + step]))
                          for i in range(0, len(normalized_images), step)], dim=0)

    def pool_features(self, features: torch.Tensor, slots: torch.Tensor, mask: torch.Tensor,
                      positions: torch.Tensor) -> torch.Tensor:
        """B×K×12×D -> B×12, with the inherited slot/position gated MIL formula."""
        if features.ndim != 4 or features.shape[2:] != (len(TARGETS), self.feature_dim):
            raise ValueError('Expected B×K×12×D class-specific window features')
        if slots.shape != features.shape[:2] or mask.shape != slots.shape or positions.shape != slots.shape:
            raise ValueError('Bag metadata shape mismatch')
        valid = mask.bool()
        if not valid.any(1).all():
            raise ValueError('All-invalid study bag is not a prediction')
        if ((slots < 0) | (slots > 5)).any() or not torch.isfinite(positions).all():
            raise ValueError('Invalid slot or position metadata')
        h = self.norm(features.float())
        embeddings = self.slot(slots.long()) + self.position(positions.float().unsqueeze(-1))
        h = h + embeddings[:, :, None, :]
        # One dropout mask per window/feature, shared over class readouts. At
        # uniform pooling this is the ordinary head's B×K×D dropout operation.
        if self.training and self.dropout.p:
            h = h * self.dropout(torch.ones_like(h[:, :, 0, :]))[:, :, None, :]
        gated = torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))
        scores = torch.einsum('bkcd,cd->bkc', gated, self.att_out.weight) + self.att_out.bias
        scores = scores.float().masked_fill(~valid[:, :, None], float('-inf'))
        attention = scores.softmax(dim=1)
        pooled = torch.einsum('bkc,bkcd->bcd', attention, h)
        logits = (pooled * self.classifier_weight[None]).sum(-1) + self.classifier_bias
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite study logits')
        return logits

    def forward(self, windows: torch.Tensor, slots: torch.Tensor, mask: torch.Tensor,
                positions: torch.Tensor, augment: bool = False, chunk: int = 0) -> torch.Tensor:
        if mask.ndim != 2 or not mask.bool().any(1).all():
            raise ValueError('All-invalid study bag is not a prediction')
        x = preprocess_windows(windows, self.config, augment)
        b, k = x.shape[:2]
        if mask.shape != (b, k):
            raise ValueError('Window mask shape mismatch')
        valid_features = self.encode_windows(x[mask.bool()], chunk)
        features = valid_features.new_zeros((b, k, len(TARGETS), self.feature_dim))
        features[mask.bool()] = valid_features
        return self.pool_features(features, slots, mask, positions)


def paired_models(config: dict[str, Any], *, backbone: nn.Module | None = None,
                  pretrained: bool = False) -> tuple[SpatialTargetMIL, SpatialTargetMIL]:
    """Identical initial bytes for both arms; only the spatial path is enabled."""
    control = SpatialTargetMIL(config, False, backbone=backbone, pretrained=pretrained)
    spatial = SpatialTargetMIL(config, True, backbone=copy.deepcopy(control.backbone), pretrained=False)
    spatial.load_state_dict(control.state_dict(), strict=True)
    return control, spatial


def save_checkpoint(model: SpatialTargetMIL, path: str | Path, *, contract: dict[str, Any],
                    extras: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    payload = {'schema': 'rsna_spatial_ablation_v1', 'config': model.config,
               'spatial': model.spatial, 'contract': contract,
               'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extras:
        if set(extras) & set(payload):
            raise ValueError('Extras overwrite checkpoint contract')
        payload.update(extras)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(model: SpatialTargetMIL, path: str | Path, *, contract: dict[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'rsna_spatial_ablation_v1' or payload.get('config') != model.config \
            or payload.get('spatial') != model.spatial or payload.get('contract') != contract:
        raise ValueError('Checkpoint architecture, arm, configuration, or split contract differs')
    model.load_state_dict(payload['state_dict'], strict=True)
    return payload
