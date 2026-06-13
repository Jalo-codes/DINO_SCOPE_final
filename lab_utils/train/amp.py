"""AMP settings resolver based on hardware capabilities."""

import torch

def resolve_amp(device, want_amp: bool = True) -> tuple[bool, torch.dtype | None]:
    """Resolves mixed-precision settings based on hardware support.

    - If want_amp is False or device is not CUDA, AMP is disabled (False, None).
    - If CUDA compute capability >= 8.0, uses bfloat16 (True, torch.bfloat16).
    - If CUDA compute capability < 8.0, uses float16 (True, torch.float16).
    """
    if not want_amp:
        return False, None

    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    if device_obj.type != 'cuda' or not torch.cuda.is_available():
        return False, None

    try:
        cc_major, cc_minor = torch.cuda.get_device_capability(device_obj)
        cc = cc_major + cc_minor / 10.0
    except Exception:
        cc = 0.0

    if cc >= 8.0:
        return True, torch.bfloat16
    elif cc > 0.0:
        return True, torch.float16
    else:
        return False, None
