"""lab_utils.model.image_bce_detector — DINOv3 + LoRA + attention pool for image-level BCE.

Pure image-level binary classifier: is this image forged?
No patch-level outputs, no contrastive heads, no localization.

Architecture:
    DINOv3 (frozen) + LoRA → patch features (B, N, D)
    → Gated attention pool (Ilse et al. 2018 MIL) → (B, D)
    → Linear → (B,) logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from lab_utils.errors import DataError
from lab_utils.data.resolution import Resolution


class AttentionPool(nn.Module):
    """Gated attention pooling: (B, N, D) → (B,) image logit.

    Implements the gated-attention MIL mechanism from Ilse et al. 2018:
        a_i = softmax( w^T ( tanh(Vz_i) ⊙ sigmoid(Uz_i) ) )
        z   = sum_i a_i * z_i
        logit = w_out^T z
    """

    def __init__(self, d_in: int, d_hidden: int = 256):
        super().__init__()
        self.V   = nn.Linear(d_in, d_hidden, bias=False)
        self.U   = nn.Linear(d_in, d_hidden, bias=False)
        self.w   = nn.Linear(d_hidden, 1, bias=False)
        self.out = nn.Linear(d_in, 1, bias=True)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, N, D)
        h = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))  # (B, N, d_hidden)
        a = self.w(h).softmax(dim=1)                           # (B, N, 1)
        pooled = (a * x).sum(dim=1)                            # (B, D)
        logit = self.out(pooled).squeeze(-1)                   # (B,)
        if return_attention:
            return logit, a.squeeze(-1)                        # (B,), (B, N)
        return logit


class ImageBCEDetector(nn.Module):
    """DINOv3 + LoRA backbone with a gated attention pool for image-level BCE.

    Args:
        model_name:   HuggingFace model id (same as ContrastiveDetector).
        res:          Resolution — used for input shape assertion and patch slicing.
        lora_rank:    LoRA rank.
        lora_alpha:   LoRA alpha.
        lora_dropout: LoRA dropout.
        lora_targets: Substring patterns to select LoRA target modules.
        pool_hidden:  Hidden dim of the attention pool MLP.
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
        pool_hidden: int = 256,
    ):
        super().__init__()
        self.res = res

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
        self.pool = AttentionPool(feat_dim, d_hidden=pool_hidden)

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, num_patches, hidden_dim)."""
        out = self.backbone(pixel_values=x).last_hidden_state
        return out[:, -self.res.num_patches:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B,) image-level logit. Positive = splice."""
        if x.dim() != 4:
            raise DataError(
                f"ImageBCEDetector.forward: expected 4D input (B, 3, H, W), got {tuple(x.shape)}"
            )
        _, C, H, W = x.shape
        expected = self.res.image_size
        if C != 3 or H != expected or W != expected:
            raise DataError(
                f"ImageBCEDetector.forward: expected (B, 3, {expected}, {expected}), "
                f"got (B, {C}, {H}, {W})"
            )
        return self.pool(self.encode_patches(x))


def build_image_bce_detector(
    *,
    model_name: str,
    resolution: Resolution,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                           'up_proj', 'down_proj'),
    pool_hidden: int = 256,
    device=None,
) -> ImageBCEDetector:
    model = ImageBCEDetector(
        model_name=model_name,
        res=resolution,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_targets=lora_targets,
        pool_hidden=pool_hidden,
    )
    if device is not None:
        model = model.to(device)
    return model
