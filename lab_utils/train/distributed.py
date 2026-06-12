"""lab_utils.train.distributed — minimal DDP helpers.

Provides a DistributedContext and thin wrappers so training code doesn't
import torch.distributed directly.  Also exposes work-sharding utilities
for diagnose / eval scripts that want to fan out across ranks (one cell of
a sweep per rank, gather results on rank 0).
"""

import dataclasses
import os
from datetime import timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from lab_utils.logging.text import log_line


@dataclasses.dataclass
class DistributedContext:
    """Runtime distributed-training state."""
    is_distributed: bool
    rank:           int
    world_size:     int
    local_rank:     int
    is_main:        bool   # True iff rank == 0


def setup(backend: str = 'nccl') -> DistributedContext:
    """Initialise the process group (or return a single-process context).

    Reads RANK, LOCAL_RANK, WORLD_SIZE from the environment (set by
    torchrun / torch.distributed.launch).  Falls back to non-distributed
    if the env vars are absent.

    Returns:
        DistributedContext
    """
    rank       = int(os.environ.get('RANK',       0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        if local_rank >= n_devices:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {n_devices} CUDA device(s) are visible. "
                "Check CUDA_VISIBLE_DEVICES and --nproc_per_node."
            )
        torch.cuda.set_device(local_rank)

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(hours=4))
        log_line(f'[dist] initialized rank={rank} world_size={world_size}')

    ctx = DistributedContext(
        is_distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=(rank == 0),
    )
    return ctx


def barrier(ctx: Optional[DistributedContext] = None) -> None:
    """Synchronise all ranks."""
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    """Destroy the process group if initialized — clean DDP shutdown.

    Call once at the very end of a distributed run (normal completion AND in a
    finally on crash). Without it you get the
    'destroy_process_group() was not called ... can leak resources' warning, and
    — worse for back-to-back runs — the NCCL communicator + rendezvous linger,
    so the NEXT torchrun in a sweep loop can fail to start (or torchrun returns
    non-zero and breaks the loop). No-op when single-process.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks in-place."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(dist.get_world_size())
    return tensor


def broadcast_scalar(value: float, device: torch.device, src: int = 0) -> float:
    """Broadcast a scalar float from `src` to all ranks."""
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return float(tensor.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when model is DDP-wrapped."""
    return model.module if hasattr(model, 'module') else model


def wrap_model(
    model: torch.nn.Module,
    ctx: DistributedContext,
    *,
    device: Optional[torch.device] = None,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> torch.nn.Module:
    """Wrap a model in DistributedDataParallel when ctx is distributed."""
    if not ctx.is_distributed:
        return model
    if device is None:
        device = torch.device(f'cuda:{ctx.local_rank}' if torch.cuda.is_available() else 'cpu')
    ddp_kwargs = {'find_unused_parameters': bool(find_unused_parameters)}
    if device.type == 'cuda':
        ddp_kwargs.update(device_ids=[ctx.local_rank], output_device=ctx.local_rank)
    wrapped = DistributedDataParallel(model, **ddp_kwargs)
    if static_graph and hasattr(wrapped, '_set_static_graph'):
        wrapped._set_static_graph()
        log_line(f'[dist] enabled DDP static_graph rank={ctx.rank}')
    log_line(f'[dist] wrapped model in DDP rank={ctx.rank} local_rank={ctx.local_rank}')
    return wrapped


# ---------------------------------------------------------------------------
# Work-sharding helpers (for diagnose / eval sweeps over independent items)
# ---------------------------------------------------------------------------


def shard_iterable(
    items: Sequence,
    ctx: Optional[DistributedContext] = None,
) -> List:
    """Return rank-strided slice ``items[rank::world_size]``.

    Pure function: no torch.distributed calls.  When ``ctx`` is ``None`` or
    not distributed, returns ``list(items)`` unchanged so single-GPU and
    DDP entry points share the same code path.
    """
    if ctx is None or not ctx.is_distributed:
        return list(items)
    return list(items)[ctx.rank::ctx.world_size]


def gather_objects(
    obj: Any,
    ctx: Optional[DistributedContext] = None,
) -> List[Any]:
    """Gather one Python object per rank onto every rank.

    On rank 0, returns ``[obj_rank0, obj_rank1, ...]``; on other ranks the
    same list is returned.  Single-process fallback returns ``[obj]``.
    Useful for collecting per-cell metric dicts into a single rank-0 view
    for logging.

    All objects must be pickleable (``all_gather_object`` constraint).
    """
    if ctx is None or not ctx.is_distributed or not dist.is_initialized():
        return [obj]
    out: List[Any] = [None] * ctx.world_size
    dist.all_gather_object(out, obj)
    return out


def gather_dicts(
    local: Dict[Any, Any],
    ctx: Optional[DistributedContext] = None,
    *,
    merge: str = 'first_wins',
) -> Dict[Any, Any]:
    """Merge per-rank dicts into one dict (visible to all ranks).

    Args:
        local:  Dict produced by this rank (e.g. ``{cell_id: metrics}``).
        ctx:    Distributed context; ``None`` -> identity.
        merge:  ``'first_wins'`` keeps the lowest-rank entry on collision;
                ``'last_wins'`` keeps the highest.  When sweep cells are
                disjoint across ranks (the expected pattern), no collisions
                arise and the choice is irrelevant.

    Single-process fallback returns ``dict(local)``.
    """
    if ctx is None or not ctx.is_distributed or not dist.is_initialized():
        return dict(local)
    pieces = gather_objects(local, ctx)
    merged: Dict[Any, Any] = {}
    if merge == 'last_wins':
        for piece in pieces:
            if isinstance(piece, dict):
                merged.update(piece)
    else:  # 'first_wins'
        for piece in pieces:
            if isinstance(piece, dict):
                for k, v in piece.items():
                    merged.setdefault(k, v)
    return merged
