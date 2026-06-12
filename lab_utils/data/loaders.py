"""lab_utils.data.loaders — small DataLoader factories for LabDataset batches."""

import dataclasses
from typing import Optional, Sequence

import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from lab_utils.data.dataset import lab_collate_fn


@dataclasses.dataclass(frozen=True)
class LoaderConfig:
    """Explicit DataLoader knobs shared by training and eval scripts."""
    batch_size: int
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    train_samples_per_epoch: Optional[int] = None
    sample_weights: Optional[Sequence[float]] = None
    distributed: bool = False
    rank: int = 0
    world_size: int = 1


class DistributedRandomSubsetSampler(Sampler):
    """DDP sampler with a hard global sample cap per epoch.

    `train_samples_per_epoch` is interpreted as a global cap across all ranks.
    Each rank receives a deterministic shard, and `set_epoch()` changes the
    shuffle order in the same way PyTorch's DistributedSampler does.
    """

    def __init__(
        self,
        dataset,
        *,
        num_samples: int,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.global_num_samples = max(1, int(num_samples))
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = (self.global_num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))
        indices = indices[:min(self.global_num_samples, n)]
        if len(indices) < self.total_size:
            repeats = (self.total_size + len(indices) - 1) // max(1, len(indices))
            indices = (indices * repeats)[:self.total_size]
        else:
            indices = indices[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class DistributedWeightedSampler(Sampler):
    """DDP sampler that draws weighted samples with replacement.

    `num_samples` is interpreted as a global target across all ranks. Sampling
    is deterministic per epoch so every rank sees a disjoint shard of the same
    globally sampled index list.
    """

    def __init__(
        self,
        dataset,
        *,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        if len(weights) != len(dataset):
            raise ValueError(
                f"DistributedWeightedSampler: len(weights)={len(weights)} "
                f"!= len(dataset)={len(dataset)}"
            )
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if float(self.weights.sum().item()) <= 0:
            raise ValueError("DistributedWeightedSampler: weights must sum to > 0.")
        self.global_num_samples = max(1, int(num_samples))
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = (self.global_num_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.global_num_samples,
            replacement=True,
            generator=g,
        ).tolist()
        if len(indices) < self.total_size:
            repeats = (self.total_size + len(indices) - 1) // max(1, len(indices))
            indices = (indices * repeats)[:self.total_size]
        else:
            indices = indices[:self.total_size]
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_train_loader(dataset, cfg: LoaderConfig) -> DataLoader:
    """Build a randomized LabDataset loader.

    Args:
        dataset: LabDataset or compatible dataset returning lab sample dicts.
        cfg: LoaderConfig with explicit batch/worker/memory settings.
    """
    if cfg.sample_weights is not None:
        if cfg.distributed and int(cfg.world_size) > 1:
            sampler = DistributedWeightedSampler(
                dataset,
                weights=cfg.sample_weights,
                num_samples=int(cfg.train_samples_per_epoch or len(dataset)),
                num_replicas=int(cfg.world_size),
                rank=int(cfg.rank),
            )
        else:
            sampler = WeightedRandomSampler(
                torch.as_tensor(cfg.sample_weights, dtype=torch.double),
                num_samples=int(cfg.train_samples_per_epoch or len(dataset)),
                replacement=True,
            )
    elif cfg.distributed and int(cfg.world_size) > 1:
        if cfg.train_samples_per_epoch:
            sampler = DistributedRandomSubsetSampler(
                dataset,
                num_samples=int(cfg.train_samples_per_epoch),
                num_replicas=int(cfg.world_size),
                rank=int(cfg.rank),
                shuffle=True,
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=int(cfg.world_size),
                rank=int(cfg.rank),
                shuffle=True,
                drop_last=bool(cfg.drop_last),
            )
    elif cfg.train_samples_per_epoch:
        sampler = RandomSampler(dataset, num_samples=int(cfg.train_samples_per_epoch))
    else:
        sampler = RandomSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        collate_fn=lab_collate_fn,
        drop_last=bool(cfg.drop_last),
    )


def build_eval_loader(dataset, cfg: LoaderConfig) -> DataLoader:
    """Build a deterministic sequential LabDataset loader."""
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=int(cfg.world_size),
            rank=int(cfg.rank),
            shuffle=False,
            drop_last=False,
        )
        if cfg.distributed and int(cfg.world_size) > 1
        else SequentialSampler(dataset)
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        collate_fn=lab_collate_fn,
        drop_last=bool(cfg.drop_last),
    )

__all__ = [
    'LoaderConfig',
    'build_train_loader',
    'build_eval_loader',
    'DistributedWeightedSampler',
]
