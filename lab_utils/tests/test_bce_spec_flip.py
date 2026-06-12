"""Tests for the IMD2020BCESpec train/val routing, incl. the train-on-CASIA /
val-on-IMD flip (``imd_train=False, casia_train=True``).

The indexers are monkeypatched to return synthetic items, so this exercises the
routing logic only — no disk, no images.
"""

import types

import contrastive_inpainting_v1.experiments.imd2020_bce as bce_spec
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec


_CFG = types.SimpleNamespace(valid_exts=('.jpg', '.png'))


def _fake_imd(*_a, **_k):
    train = [{'source': 'imd2020', 'id': 'imd_t0'}, {'source': 'imd2020', 'id': 'imd_t1'}]
    val   = [{'source': 'imd2020', 'id': 'imd_v0'}]
    return train, val


def _fake_casia(*_a, **_k):
    train = [{'source': 'casia', 'id': 'cas_t0'}, {'source': 'casia', 'id': 'cas_t1'}]
    val   = [{'source': 'casia', 'id': 'cas_v0'}]
    return train, val


def _patch(monkeypatch):
    monkeypatch.setattr(bce_spec, 'index_imd2020', _fake_imd)
    monkeypatch.setattr(bce_spec, 'index_casia_exported', _fake_casia)


def _ids(items, source):
    return {it['id'] for it in items if it['source'] == source}


def test_flip_train_casia_val_imd(monkeypatch):
    _patch(monkeypatch)
    spec = IMD2020BCESpec(
        imd2020_root='x', casia_root='y',
        imd_train=False, casia_train=True,
    )
    train, val = spec.build_items(_CFG)

    # No IMD anywhere in train; every IMD item (train+val halves) is held out.
    assert _ids(train, 'imd2020') == set()
    assert _ids(val, 'imd2020') == {'imd_t0', 'imd_t1', 'imd_v0'}

    # CASIA train half trains; CASIA val half validates (in-domain headline).
    assert _ids(train, 'casia') == {'cas_t0', 'cas_t1'}
    assert _ids(val, 'casia') == {'cas_v0'}


def test_default_train_imd_val_casia(monkeypatch):
    _patch(monkeypatch)
    spec = IMD2020BCESpec(imd2020_root='x', casia_root='y')  # imd_train=True, casia_train=False
    train, val = spec.build_items(_CFG)

    # IMD trains on its train half, validates on its val half.
    assert _ids(train, 'imd2020') == {'imd_t0', 'imd_t1'}
    assert _ids(val, 'imd2020') == {'imd_v0'}

    # No CASIA in train; all CASIA held out as val.
    assert _ids(train, 'casia') == set()
    assert _ids(val, 'casia') == {'cas_t0', 'cas_t1', 'cas_v0'}


def test_imd_val_only_without_casia_train_leaves_train_without_casia(monkeypatch):
    # Defensive: imd_val_only alone (casia still val-only) yields an EMPTY train
    # set from these two sources — the run script must pair the two flags.
    _patch(monkeypatch)
    spec = IMD2020BCESpec(imd2020_root='x', casia_root='y', imd_train=False)
    train, val = spec.build_items(_CFG)
    assert _ids(train, 'imd2020') == set()
    assert _ids(train, 'casia') == set()
    assert len(train) == 0
