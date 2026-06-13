#!/usr/bin/env python3
"""inspect_data.py — standalone integrity sweep over the training/val item mix.

Builds the EXACT item list the trainer would load (via IMD2020BCESpec.build_items,
the same indexers), then opens every image / mask / pasted-original the way
LabDataset.__getitem__ will, and reports anything that would either crash the
run or enter training silently mislabeled.

It does not import torch-heavy training code beyond the indexers, and it only
READS files. Nothing is modified.

Usage (mirror your train flags — only pass the roots you actually train on):

    python inspect_data.py \
        --casia_root /content/casia \
        --coco_inpaint_root /content/inpaint_coco/ \
        --sagid_root /content/sagi_d_partial \
        --bfree_root /content/B-Free-Subset-9k \
        --anyedit_root /content/AnyEdit \
        --imd2020_root /content/IMD2020 \
        --imd_val_only --casia_train \
        --report bad_items.jsonl

Checks per item:
  MISSING_IMG        img path not on disk
  UNREADABLE_IMG     PIL could not open/decode (corrupt / truncated / not an image)
  DEGENERATE_SIZE    image smaller than --min_side on either axis
  SPLICE_NO_MASK     splice-kind item carries no mask path  -> fake label, no region
  MISSING_MASK       mask path set but not on disk          -> crashes at load
  UNREADABLE_MASK    mask present but PIL could not decode it
  EMPTY_MASK         splice mask is all-zero                -> fake label, empty target
  MASK_SIZE_MISMATCH mask size != img size (trainer NEAREST-coerces -> silent misalign)
  MISSING_REALPATH   real_path set but not on disk          -> paste path crashes
  UNREADABLE_REALPATH real_path present but PIL could not decode it

Exit code is non-zero if any HARD issue (crash-class) is found.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# Keep PIL strict: truncated files should RAISE here exactly as they will in
# training (LabDataset does not enable LOAD_TRUNCATED_IMAGES).
Image.MAX_IMAGE_PIXELS = None  # silence DecompressionBomb warnings; we measure size ourselves

# Repo root on path so `lab_utils` / `contrastive_inpainting_v1` import cleanly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contrastive_inpainting_v1.configs.base import Config            # noqa: E402
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec  # noqa: E402

_SPLICE_KINDS = frozenset({'imd_splice', 'casia_splice'})

# Hard = would crash the run or feed a contradictory label; Soft = smell/coerced.
_HARD = {'MISSING_IMG', 'UNREADABLE_IMG', 'SPLICE_NO_MASK', 'MISSING_MASK',
         'UNREADABLE_MASK', 'EMPTY_MASK', 'MISSING_REALPATH', 'UNREADABLE_REALPATH'}
_SOFT = {'DEGENERATE_SIZE', 'MASK_SIZE_MISMATCH'}


def _open_size(path):
    """Open + fully decode an image. Returns (w, h). Raises on corrupt/truncated."""
    with Image.open(path) as im:
        im.load()              # force full decode -> truncated files raise here
        im.convert('RGB')      # mirror the loader's convert path
        return im.size


def _check_item(item, min_side):
    """Return list of (issue, detail) for one item. Empty list = clean."""
    issues = []
    kind = str(item.get('kind', ''))
    is_splice = kind in _SPLICE_KINDS
    img = item.get('img')
    mask = item.get('mask')
    real_path = item.get('real_path')

    # ── image ──
    img_size = None
    if not img or not os.path.exists(img):
        issues.append(('MISSING_IMG', img))
    else:
        try:
            img_size = _open_size(img)
            if min(img_size) < min_side:
                issues.append(('DEGENERATE_SIZE', f'{img_size}'))
        except Exception as exc:
            issues.append(('UNREADABLE_IMG', f'{type(exc).__name__}: {exc}'))

    # ── mask ──
    if is_splice and not mask:
        issues.append(('SPLICE_NO_MASK', kind))
    elif mask:
        if not os.path.exists(mask):
            issues.append(('MISSING_MASK', mask))
        else:
            try:
                with Image.open(mask) as mk:
                    mk.load()
                    m = np.asarray(mk.convert('L'), dtype=np.uint8)
                    mask_size = mk.size
                if is_splice and int(m.max()) == 0:
                    issues.append(('EMPTY_MASK', 'all-zero mask on splice'))
                if img_size is not None and mask_size != img_size:
                    issues.append(('MASK_SIZE_MISMATCH', f'img={img_size} mask={mask_size}'))
            except Exception as exc:
                issues.append(('UNREADABLE_MASK', f'{type(exc).__name__}: {exc}'))

    # ── pasted original (only loaded when present) ──
    if real_path:
        if not os.path.exists(real_path):
            issues.append(('MISSING_REALPATH', real_path))
        else:
            try:
                _open_size(real_path)
            except Exception as exc:
                issues.append(('UNREADABLE_REALPATH', f'{type(exc).__name__}: {exc}'))

    return issues


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--coco_inpaint_root', default=None)
    p.add_argument('--sagid_root', default=None)
    p.add_argument('--bfree_root', default=None)
    p.add_argument('--anyedit_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--imd_val_only', action='store_true',
                   help='IMD held out as OOD val (matches training flag).')
    p.add_argument('--casia_train', action='store_true',
                   help='CASIA in train (matches training flag).')
    p.add_argument('--split', choices=('train', 'val', 'both'), default='train',
                   help='Which item set to validate (default: train).')
    p.add_argument('--min_side', type=int, default=32,
                   help='Flag images whose shorter side is below this (DEGENERATE_SIZE).')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0,
                   help='Validate only the first N items per split (0 = all).')
    p.add_argument('--report', default='bad_items.jsonl',
                   help='Write one JSON line per flagged item here.')
    return p


def main():
    args = _build_parser().parse_args()
    cfg = Config()

    spec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root,
        casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        coco_inpaint_root=args.coco_inpaint_root,
        sagid_root=args.sagid_root,
        bfree_root=args.bfree_root,
        anyedit_root=args.anyedit_root,
        imd_train=not args.imd_val_only,
        casia_train=args.casia_train,
    )
    train_items, val_items = spec.build_items(cfg)

    sets = {'train': train_items, 'val': val_items}
    chosen = ['train', 'val'] if args.split == 'both' else [args.split]

    items = []
    for s in chosen:
        si = sets[s]
        if args.limit:
            si = si[:args.limit]
        for it in si:
            it = dict(it)
            it['_split'] = s
            items.append(it)

    print(f'\n=== inspect_data: validating {len(items)} items '
          f'(split={args.split}, workers={args.workers}) ===', flush=True)

    # Composition snapshot (cross-check against the training log).
    by_src = Counter(str(i.get('source', '?')) for i in items)
    by_kind = Counter(str(i.get('kind', '?')) for i in items)
    print(f'  sources: {dict(sorted(by_src.items()))}')
    print(f'  kinds:   {dict(sorted(by_kind.items()))}\n', flush=True)

    t0 = time.time()
    flagged = []                       # (item, issues)
    issue_counts = Counter()
    issue_by_source = defaultdict(Counter)
    done = 0

    def work(it):
        return it, _check_item(it, args.min_side)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for it, issues in ex.map(work, items):
            done += 1
            if issues:
                flagged.append((it, issues))
                src = str(it.get('source', '?'))
                for code, _detail in issues:
                    issue_counts[code] += 1
                    issue_by_source[src][code] += 1
            if done % 2000 == 0:
                print(f'  ...{done}/{len(items)} checked '
                      f'({len(flagged)} flagged so far)', flush=True)

    elapsed = time.time() - t0

    # ── report file ──
    n_written = 0
    if flagged:
        with open(args.report, 'w') as fh:
            for it, issues in flagged:
                fh.write(json.dumps({
                    'split': it.get('_split'),
                    'source': it.get('source'),
                    'kind': it.get('kind'),
                    'img': it.get('img'),
                    'mask': it.get('mask'),
                    'real_path': it.get('real_path'),
                    'issues': [{'code': c, 'detail': d} for c, d in issues],
                }) + '\n')
                n_written += 1

    # ── summary ──
    print(f'\n=== summary ({elapsed:.1f}s) ===')
    print(f'  items checked : {len(items)}')
    print(f'  items flagged : {len(flagged)}')
    n_hard = sum(1 for _it, iss in flagged if any(c in _HARD for c, _ in iss))
    n_soft_only = len(flagged) - n_hard
    print(f'    hard (crash/bad-label): {n_hard}')
    print(f'    soft (coerced/smell)  : {n_soft_only}')

    if issue_counts:
        print('\n  issue breakdown (item-issue pairs):')
        for code, n in issue_counts.most_common():
            tag = 'HARD' if code in _HARD else ('soft' if code in _SOFT else '?')
            print(f'    {code:<20} {n:>7}  [{tag}]')

        print('\n  by source:')
        for src in sorted(issue_by_source):
            parts = ', '.join(f'{c}={n}' for c, n in issue_by_source[src].most_common())
            print(f'    {src:<14} {parts}')

    if n_written:
        print(f'\n  wrote {n_written} flagged items → {args.report}')
        print('  (inspect with:  head -5 ' + args.report + '  |  python -m json.tool)')

    if n_hard:
        print('\n  RESULT: HARD issues present — these crash the run or feed bad '
              'labels. Cull them before training.')
        sys.exit(1)
    elif flagged:
        print('\n  RESULT: only soft issues (trainer coerces these) — review but '
              'not blocking.')
        sys.exit(0)
    else:
        print('\n  RESULT: clean. No bad images found.')
        sys.exit(0)


if __name__ == '__main__':
    main()
