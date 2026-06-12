"""contrastive_inpainting_v1.experiments.tgif2_flux — TGIF2 FLUX OOD eval items.

Builds LabDataset-compatible item dicts from the pre-normalized
``tgif2_index.json`` (produced by the Colab pipeline; per-coco_id →
{category, original_512, masks{type_res}, manipulations[...]}).

This is a pure *out-of-distribution* probe: the flip model never trained on
diffusion inpainting. We reuse kind='imd_splice'/'imd_real' so the existing
localization eval (which gates on those kinds) accepts the items unchanged —
the TGIF provenance rides along in ``tgif_*`` tag fields the caller uses to
slice results (the headline axis is sp vs fr — paste-back vs full re-encode).

Item dict (LabDataset reads img/mask/kind/case_id/source; tgif_* are for us):
    {'img','mask','kind','case_id','source','tgif_type','tgif_model',
     'tgif_mask_type','tgif_mask_family','tgif_coco_id','tgif_category',
     'tgif_var_id'}
"""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.logging.text import log_line


SP, FR = 'sp', 'fr'
MASK_TYPES = ('bbox', 'segm', 'random')


def _mask_type_short(mask_used: str) -> str:
    """'bbox_512' -> 'bbox', 'random_512' -> 'random', etc."""
    return str(mask_used).split('_', 1)[0]


def _mask_family(mask_type: str) -> str:
    return 'random' if mask_type == 'random' else 'semantic'


def cell_key(model: str, type_: str, mask_type: str) -> Tuple[str, str, str]:
    return (model, type_, mask_type)


def split_tgif2_coco_ids(
    index_path: str,
    *,
    train_frac: float = 0.5,
    seed: str = 'tgif_fr_half',
) -> Tuple[List[str], List[str]]:
    """Deterministic (train_ids, eval_ids) partition of the index's coco_ids.

    Splitting at the coco_id level keeps every manipulation variant AND the
    pristine original of a source image on the same side — no content leakage
    between a fine-tune half and its held-out eval half. Same md5(seed|key)
    ranking idiom as ``deterministic_subsample``, so the split is stable
    across runs and machines.
    """
    with open(index_path) as f:
        index = json.load(f)
    ids = sorted(index.keys())
    ranked = sorted(
        ids, key=lambda cid: hashlib.md5(f'{seed}|{cid}'.encode('utf-8')).hexdigest()
    )
    n_train = int(round(len(ranked) * train_frac))
    return sorted(ranked[:n_train]), sorted(ranked[n_train:])


def build_tgif2_items(
    root: str,
    index_path: Optional[str] = None,
    *,
    max_per_cell: Optional[int] = None,
    include_reals: bool = True,
    seed: str = 'tgif2',
    coco_ids: Optional[Set[str]] = None,
    types: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (fake_items, real_items) from tgif2_index.json.

    Args:
        root:          Dataset root (the dir that index paths are relative to,
                       e.g. .../content/flux_originals).
        index_path:    Path to tgif2_index.json (default <root>/tgif2_index.json).
        max_per_cell:  If set, deterministically subsample each
                       (model, type, mask_type) cell to this many fakes.
        include_reals: Also build one real item per coco_id (the COCO original).
        seed:          Base seed for the per-cell subsample.
        coco_ids:      If set, only entries whose coco_id is in this set are
                       used (fakes AND reals) — half-split fine-tune support.
        types:         If set, only manipulations of these types ('sp'/'fr').
    """
    index_path = index_path or os.path.join(root, 'tgif2_index.json')
    with open(index_path) as f:
        index = json.load(f)

    by_cell: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    real_items: List[Dict[str, Any]] = []
    n_missing_fake = n_missing_mask = 0

    for coco_id, entry in index.items():
        if coco_ids is not None and str(coco_id) not in coco_ids:
            continue
        category = entry.get('category', '')
        masks = entry.get('masks', {})

        if include_reals:
            orig = entry.get('original_512')
            if orig:
                real_path = os.path.join(root, orig)
                if os.path.exists(real_path):
                    real_items.append({
                        'img': real_path, 'mask': None, 'kind': 'imd_real',
                        'case_id': str(coco_id), 'source': 'tgif2',
                        'tgif_coco_id': str(coco_id), 'tgif_category': category,
                    })

        for man in entry.get('manipulations', []):
            if types is not None and man.get('type', '') not in types:
                continue
            fake_path = os.path.join(root, man['fake_path'])
            mask_used = man.get('mask_used', '')
            mask_rel = masks.get(mask_used)
            if not mask_rel:
                n_missing_mask += 1
                continue
            mask_path = os.path.join(root, mask_rel)
            if not os.path.exists(fake_path):
                n_missing_fake += 1
                continue
            if not os.path.exists(mask_path):
                n_missing_mask += 1
                continue
            model = man.get('model', '')
            type_ = man.get('type', '')
            mtype = _mask_type_short(mask_used)
            var_id = int(man.get('variation_id', 0))
            by_cell[cell_key(model, type_, mtype)].append({
                'img': fake_path, 'mask': mask_path, 'kind': 'imd_splice',
                'case_id': f'{coco_id}_{model}_{type_}_{mtype}_v{var_id}',
                'source': 'tgif2',
                'tgif_type': type_, 'tgif_model': model,
                'tgif_mask_type': mtype, 'tgif_mask_family': _mask_family(mtype),
                'tgif_coco_id': str(coco_id), 'tgif_category': category,
                'tgif_var_id': var_id,
            })

    fake_items: List[Dict[str, Any]] = []
    for ck in sorted(by_cell):
        bucket = by_cell[ck]
        if max_per_cell is not None and len(bucket) > max_per_cell:
            bucket = deterministic_subsample(
                bucket, max_per_cell, seed=f'{seed}:{ck[0]}:{ck[1]}:{ck[2]}'
            )
        fake_items.extend(bucket)

    log_line(
        f'[data] tgif2_flux: fakes={len(fake_items)} reals={len(real_items)} '
        f'cells={len(by_cell)} max_per_cell={max_per_cell} '
        f'missing_fake={n_missing_fake} missing_mask={n_missing_mask}'
        + (f' coco_id_filter={len(coco_ids)}' if coco_ids is not None else '')
        + (f' types={sorted(types)}' if types is not None else '')
    )
    return fake_items, real_items


# ── dry-run: print per-cell counts + path validity, no model needed ──────────

def _dry_run(root: str, index_path: Optional[str], max_per_cell: Optional[int]) -> None:
    fakes, reals = build_tgif2_items(
        root, index_path, max_per_cell=max_per_cell, include_reals=True
    )
    by_type = Counter(f['tgif_type'] for f in fakes)
    by_model = Counter(f['tgif_model'] for f in fakes)
    by_mtype = Counter(f['tgif_mask_type'] for f in fakes)
    cells = Counter((f['tgif_model'], f['tgif_type'], f['tgif_mask_type']) for f in fakes)

    print(f'root={root}')
    print(f'fakes={len(fakes)}  reals={len(reals)}')
    print(f'by type:  {dict(by_type)}')
    print(f'by model: {dict(by_model)}')
    print(f'by mask:  {dict(by_mtype)}')
    print('per-cell (model, type, mask) counts:')
    for ck in sorted(cells):
        print(f'   {ck}: {cells[ck]}')
    # spot-check first item of each type resolves
    for t in (SP, FR):
        ex = next((f for f in fakes if f['tgif_type'] == t), None)
        if ex:
            print(f'sample {t}: img_ok={os.path.exists(ex["img"])} '
                  f'mask_ok={os.path.exists(ex["mask"])} case={ex["case_id"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Dry-run the TGIF2 item builder (no model).')
    ap.add_argument('--root', required=True,
                    help='e.g. /media/ssd/DINO_SCOPE_DATA/content/flux_originals')
    ap.add_argument('--index', default=None, help='default <root>/tgif2_index.json')
    ap.add_argument('--max_per_cell', type=int, default=None)
    args = ap.parse_args()
    _dry_run(args.root, args.index, args.max_per_cell)
