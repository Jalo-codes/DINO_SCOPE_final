"""Tests for lab_utils.logging.run_dir.

Verifies the resume-safe layout: stable per-experiment dir + fresh
timestamped log subdir per invocation.
"""

import datetime
import json
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.logging.run_dir import RunDir, build_run_dir


def _fixed(ts: str) -> datetime.datetime:
    return datetime.datetime.strptime(ts, '%Y%m%d-%H%M%S')


def test_creates_stable_root_and_subdirs():
    with tempfile.TemporaryDirectory() as root:
        rd = build_run_dir(root, 'exp', role='train',
                            timestamp=_fixed('20260524-120000'),
                            write_symlink=False)
        assert isinstance(rd, RunDir)
        assert rd.root.is_dir()
        assert rd.checkpoints.is_dir()
        assert rd.log_dir.is_dir()
        assert rd.artifacts_dir.is_dir()
        # log_dir name includes ts, git_sha, role
        assert rd.log_dir.name.startswith('20260524-120000_')
        assert rd.log_dir.name.endswith('_train')


def test_manifest_written_with_role_and_argv():
    with tempfile.TemporaryDirectory() as root:
        rd = build_run_dir(root, 'exp', role='train',
                            timestamp=_fixed('20260524-120000'),
                            write_symlink=False)
        assert rd.manifest_path.exists()
        with open(rd.manifest_path) as f:
            manifest = json.load(f)
        assert manifest['role'] == 'train'
        assert manifest['name'] == 'exp'
        assert 'argv' in manifest
        assert 'cwd' in manifest
        assert 'paths' in manifest


def test_resume_reuses_root_keeps_logs_separate():
    """Second invocation under the same name lands a fresh log subdir."""
    with tempfile.TemporaryDirectory() as root:
        rd1 = build_run_dir(root, 'exp', role='train',
                             timestamp=_fixed('20260524-120000'),
                             write_symlink=False)
        rd2 = build_run_dir(root, 'exp', role='eval',
                             timestamp=_fixed('20260524-120500'),
                             write_symlink=False)
        # Same stable root, same checkpoints dir
        assert rd1.root == rd2.root
        assert rd1.checkpoints == rd2.checkpoints
        # Different log dirs
        assert rd1.log_dir != rd2.log_dir
        assert rd1.log_dir.name.endswith('_train')
        assert rd2.log_dir.name.endswith('_eval')


def test_collision_shares_log_dir_for_ddp():
    """Two calls with same ts+git+role SHARE the log dir.

    This is intentional: DDP ranks call build_run_dir simultaneously and
    need a shared per-invocation dir.  The trade-off: two sequential CLI
    invocations within the same second + git + role also share the dir
    (the later caller overwrites the earlier manifest).  Acceptable edge
    case in exchange for safe DDP behavior.
    """
    with tempfile.TemporaryDirectory() as root:
        ts = _fixed('20260524-120000')
        rd1 = build_run_dir(root, 'exp', role='train',
                             timestamp=ts, write_symlink=False)
        rd2 = build_run_dir(root, 'exp', role='train',
                             timestamp=ts, write_symlink=False)
        assert rd1.log_dir == rd2.log_dir


def test_role_name_is_sanitized():
    with tempfile.TemporaryDirectory() as root:
        rd = build_run_dir(root, 'exp', role='diagnose/swin!',
                            timestamp=_fixed('20260524-120000'),
                            write_symlink=False)
        # No slashes / exclamation marks in the role suffix
        assert '/' not in rd.log_dir.name
        assert '!' not in rd.log_dir.name
        assert 'diagnose_swin' in rd.log_dir.name


def test_fspath_returns_root_for_backcompat():
    """os.path.join(rd, 'foo') still works (lands at stable root)."""
    with tempfile.TemporaryDirectory() as root:
        rd = build_run_dir(root, 'exp', role='train',
                            timestamp=_fixed('20260524-120000'),
                            write_symlink=False)
        joined = os.path.join(rd, 'checkpoint_005.pt')
        assert joined == str(rd.root / 'checkpoint_005.pt')


def test_str_returns_root_path():
    """f-string interpolation prints the stable root."""
    with tempfile.TemporaryDirectory() as root:
        rd = build_run_dir(root, 'exp', role='train',
                            timestamp=_fixed('20260524-120000'),
                            write_symlink=False)
        assert str(rd) == str(rd.root)
