"""Tests for lab_utils.train.checkpoint.find_latest_checkpoint."""

import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.train.checkpoint import find_latest_checkpoint


def _touch(path: str, mtime: int) -> None:
    open(path, 'w').close()
    os.utime(path, (mtime, mtime))


def test_returns_none_when_dir_missing():
    assert find_latest_checkpoint('/tmp/__nonexistent_dir__') is None


def test_returns_none_when_dir_empty():
    with tempfile.TemporaryDirectory() as d:
        assert find_latest_checkpoint(d) is None


def test_picks_newest_epoch_by_mtime():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, 'epoch_000.pt'), 1000)
        _touch(os.path.join(d, 'epoch_001.pt'), 1001)
        _touch(os.path.join(d, 'epoch_002.pt'), 1002)
        out = find_latest_checkpoint(d)
        assert out.endswith('epoch_002.pt')


def test_last_pt_wins_when_newer():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, 'epoch_005.pt'), 1000)
        _touch(os.path.join(d, 'last.pt'), 2000)
        out = find_latest_checkpoint(d)
        assert out.endswith('last.pt')


def test_best_pt_recognized():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, 'best.pt'), 1000)
        out = find_latest_checkpoint(d)
        assert out.endswith('best.pt')


def test_custom_patterns():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, 'epoch_000.pt'), 1000)
        # Custom pattern excludes the default — should return None.
        out = find_latest_checkpoint(d, patterns=('special_*.pt',))
        assert out is None
        # Adding the right pattern picks it up.
        _touch(os.path.join(d, 'special_xx.pt'), 1500)
        out = find_latest_checkpoint(d, patterns=('special_*.pt',))
        assert out.endswith('special_xx.pt')
