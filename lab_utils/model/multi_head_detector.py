"""lab_utils.model.multi_head_detector — DINOv3 + LoRA shared backbone with
optional image-BCE attention-pool head and optional contrastive patch-embedding
head.

A single class supports three configurations via head dims:
    BCE-only:         contrastive_dim=0, pool_hidden=256
    Contrastive-only: contrastive_dim=128, pool_hidden=0
    Joint:            contrastive_dim=128, pool_hidden=256

Reuses AttentionPool from `image_bce_detector` and mirrors its LoRA wiring so
the detection head behaves identically to the Phase-2-prime BCE detector when
`contrastive_dim=0`.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from lab_utils.errors import DataError
from lab_utils.data.resolution import Resolution
from lab_utils.model.image_bce_detector import AttentionPool


class MultiHeadDetector(nn.Module):
    """DINOv3 + LoRA backbone with optional image-BCE and contrastive heads.

    Args:
        model_name:       HuggingFace model id.
        res:              Resolution — input shape assertion + patch slicing.
        lora_rank:        LoRA rank.
        lora_alpha:       LoRA alpha.
        lora_dropout:     LoRA dropout.
        lora_targets:     Substring patterns to select LoRA target modules.
        contrastive_dim:  Output dim of the contrastive projector (0 disables).
        pool_hidden:      Hidden dim of the BCE attention pool (0 disables).
    """

    def __init__(
        self,
        model_name: str,
        res: Resolution,
        *,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                               'up_proj', 'down_proj'),
        contrastive_dim: int = 128,
        pool_hidden: int = 256,
        patch_bce: bool = False,
    ):
        super().__init__()
        if contrastive_dim <= 0 and pool_hidden <= 0 and not patch_bce:
            raise ValueError(
                'MultiHeadDetector: at least one head must be enabled '
                f'(got contrastive_dim={contrastive_dim}, pool_hidden={pool_hidden}, '
                f'patch_bce={patch_bce})'
            )
        self.res = res
        self.contrastive_dim = int(contrastive_dim)
        self.pool_hidden = int(pool_hidden)
        self.patch_bce = bool(patch_bce)

        # Shared LoRA-adapted backbone (identical wiring to ImageBCEDetector).
        base = AutoModel.from_pretrained(model_name)
        target_modules = [
            name for name, _ in base.named_modules()
            if any(s in name for s in lora_targets)
        ]
        lora_cfg = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            target_modules=target_modules,
            lora_dropout=float(lora_dropout),
            bias='none',
        )
        self.backbone = get_peft_model(base, lora_cfg)
        self.backbone.enable_input_require_grads()
        self.backbone.gradient_checkpointing_enable()
        self.backbone.print_trainable_parameters()

        feat_dim = self.backbone.config.hidden_size
        self.feat_dim = int(feat_dim)

        # Per-patch contrastive projector (L2-normalized output).
        if self.contrastive_dim > 0:
            self.contrastive_proj = nn.Linear(feat_dim, self.contrastive_dim)
        else:
            self.contrastive_proj = None

        # Image-level BCE: gated MIL attention pool.
        if self.pool_hidden > 0:
            self.pool = AttentionPool(feat_dim, d_hidden=self.pool_hidden)
        else:
            self.pool = None

        # Per-patch BCE: dense splice-flagging head (one logit per patch).
        # A single Linear, mirroring the contrastive projector, so the
        # supervised-flagging baseline differs from the contrastive head only
        # in objective, not in head capacity.
        if self.patch_bce:
            self.patch_head = nn.Linear(feat_dim, 1)
        else:
            self.patch_head = None

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, feat_dim)."""
        out = self.backbone(pixel_values=x).last_hidden_state
        return out[:, -self.res.num_patches:, :]

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """Run shared backbone once; dispatch to whichever heads are enabled.

        Returns:
            {
              'patch_feats':  (B, N, feat_dim),
              'contrastive':  (B, N, contrastive_dim)  or None,
              'image_logit':  (B,)                     or None,
              'pool_attention': (B, N) per-patch attention weights from the
                                BCE pool (sums to 1 along N), or None if no
                                BCE head. Used for cluster polarity at
                                inference: the cluster with higher mean
                                attention is the splice prediction.
              'patch_logit':  (B, N) per-patch splice logits from the dense
                                patch-BCE head, or None if disabled. Decode at
                                inference is sigmoid(patch_logit) >= threshold.
            }
        """
        if x.dim() != 4:
            raise DataError(
                f'MultiHeadDetector.forward: expected 4D input, got {tuple(x.shape)}'
            )
        _, C, H, W = x.shape
        expected = self.res.image_size
        if C != 3 or H != expected or W != expected:
            raise DataError(
                f'MultiHeadDetector.forward: expected (B, 3, {expected}, {expected}), '
                f'got (B, {C}, {H}, {W})'
            )

        patch_feats = self.encode_patches(x)              # (B, N, D)

        if self.contrastive_proj is not None:
            z = self.contrastive_proj(patch_feats)        # (B, N, d)
            z = F.normalize(z, p=2, dim=-1)               # L2-norm per patch
        else:
            z = None

        if self.pool is not None:
            image_logit, pool_attention = self.pool(patch_feats, return_attention=True)
        else:
            image_logit = None
            pool_attention = None

        if self.patch_head is not None:
            patch_logit = self.patch_head(patch_feats).squeeze(-1)   # (B, N)
        else:
            patch_logit = None

        return {
            'patch_feats': patch_feats,
            'contrastive': z,
            'image_logit': image_logit,
            'pool_attention': pool_attention,
            'patch_logit': patch_logit,
        }


def build_multi_head_detector(
    *,
    model_name: str,
    resolution: Resolution,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'up_proj', 'down_proj'),
    contrastive_dim: int = 128,
    pool_hidden: int = 256,
    patch_bce: bool = False,
    device=None,
) -> MultiHeadDetector:
    model = MultiHeadDetector(
        model_name=model_name,
        res=resolution,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        contrastive_dim=contrastive_dim,
        pool_hidden=pool_hidden,
        patch_bce=patch_bce,
    )
    if device is not None:
        model = model.to(device)
    return model
