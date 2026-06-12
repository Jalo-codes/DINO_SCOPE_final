"""Smoke tests for Step 1: imports, logging, run_dir."""

import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest

from lab_utils.errors import DataError, ConfigError, EvalError
from lab_utils.logging.text import (
    install_log, log_line, log_warn, log_error, ALLOWED_TAGS,
)
from lab_utils.logging.run_dir import build_run_dir
import lab_utils  # top-level __init__ re-exports


# ── errors ─────────────────────────────────────────────────────────────────

def test_error_types_are_distinct_exceptions():
    assert issubclass(DataError, Exception)
    assert issubclass(ConfigError, Exception)
    assert issubclass(EvalError, Exception)
    assert not issubclass(DataError, ConfigError)


# ── logging.text ────────────────────────────────────────────────────────────

def test_allowed_tags_nonempty():
    assert len(ALLOWED_TAGS) >= 9


def test_log_line_unknown_tag_raises():
    with pytest.raises(ValueError, match="must start with"):
        log_line("[unknown] bad tag")


def test_log_line_no_tag_raises():
    with pytest.raises(ValueError):
        log_line("no tag at all")


def test_log_line_writes_to_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'test.log')
        install_log(path)
        log_line('[train] hello world', echo=False)
        content = open(path).read()
        assert '[train] hello world' in content


def test_install_log_writes_header():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'run.log')
        install_log(path)
        content = open(path).read()
        assert 'log opened' in content
        assert 'pid=' in content
        assert 'git=' in content


def test_log_warn_inserts_warn():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'warn.log')
        install_log(path)
        log_warn('[eval] something suspicious', echo=False)
        content = open(path).read()
        assert 'WARN' in content


def test_log_error_inserts_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'err.log')
        install_log(path)
        log_error('[data] something broke', echo=False)
        content = open(path).read()
        assert 'ERROR' in content


def test_all_allowed_tags_pass():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'tags.log')
        install_log(path)
        for tag in sorted(ALLOWED_TAGS):
            log_line(f'{tag} test message', echo=False)


# ── logging.run_dir ─────────────────────────────────────────────────────────

def test_build_run_dir_creates_directory():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = build_run_dir(tmp, 'test_exp')
        assert os.path.isdir(run_dir)


def test_build_run_dir_writes_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = build_run_dir(tmp, 'test_exp')
        manifest_path = os.path.join(run_dir, 'manifest.json')
        assert os.path.exists(manifest_path)
        import json
        m = json.load(open(manifest_path))
        assert m['name'] == 'test_exp'
        assert 'created_at' in m
        assert 'git_sha' in m


def test_build_run_dir_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir1 = build_run_dir(tmp, 'my_run')
        run_dir2 = build_run_dir(tmp, 'my_run')
        assert run_dir1 == run_dir2


# ── top-level re-exports ────────────────────────────────────────────────────

def test_top_level_imports():
    assert hasattr(lab_utils, 'DataError')
    assert hasattr(lab_utils, 'ConfigError')
    assert hasattr(lab_utils, 'EvalError')
    assert hasattr(lab_utils, 'log_line')
    assert hasattr(lab_utils, 'log_warn')
    assert hasattr(lab_utils, 'log_error')
    assert hasattr(lab_utils, 'install_log')
