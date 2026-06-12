import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.data.loaders import DistributedWeightedSampler


class _DummyDataset:
    def __len__(self):
        return 4


def test_distributed_weighted_sampler_respects_zero_weights():
    ds = _DummyDataset()
    sampler = DistributedWeightedSampler(
        ds,
        weights=[1.0, 0.0, 0.0, 0.0],
        num_samples=6,
        num_replicas=2,
        rank=0,
        seed=123,
    )
    assert list(sampler) == [0, 0, 0]


def test_distributed_weighted_sampler_shards_global_draw():
    ds = _DummyDataset()
    s0 = DistributedWeightedSampler(
        ds,
        weights=[1.0, 1.0, 1.0, 1.0],
        num_samples=6,
        num_replicas=2,
        rank=0,
        seed=7,
    )
    s1 = DistributedWeightedSampler(
        ds,
        weights=[1.0, 1.0, 1.0, 1.0],
        num_samples=6,
        num_replicas=2,
        rank=1,
        seed=7,
    )
    i0 = list(s0)
    i1 = list(s1)
    assert len(i0) == 3
    assert len(i1) == 3
    assert any(a != b for a, b in zip(i0, i1))
