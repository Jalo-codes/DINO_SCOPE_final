"""lab_utils.data.indexer — dataset discovery helpers.

Lifted from contrastive_test/data/indexer.py.  All stdout calls replaced with
log_line('[data] ...') so output appears in structured experiment logs.
The public API is identical to the original.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

from lab_utils.logging.text import log_line


def _is_valid_image(path: str, valid_exts: tuple) -> bool:
    return os.path.splitext(path)[1].lower() in valid_exts


def _index_indoor_manifest(path: str,
                           holdout_subdir: str,
                           valid_exts: tuple) -> Tuple[List[str], List[str]]:
    train_imgs: List[str] = []
    val_imgs: List[str] = []
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('utf-8', errors='ignore')
    manifest_lines = []
    for chunk in text.replace('\x00', '\n').splitlines():
        line = chunk.strip()
        if not line or line.startswith('#'):
            continue
        manifest_lines.append(line)

    for line in manifest_lines:
        candidate = line
        if not os.path.isabs(candidate):
            candidate = os.path.join(base_dir, candidate)
        candidate = os.path.abspath(candidate)
        if not os.path.exists(candidate):
            continue
        if not _is_valid_image(candidate, valid_exts):
            continue
        path_parts = set(os.path.normpath(candidate).split(os.sep))
        if holdout_subdir and holdout_subdir in path_parts:
            val_imgs.append(candidate)
        else:
            train_imgs.append(candidate)

    if not train_imgs and not val_imgs:
        log_line('[data] MANIFEST: no valid image paths found; skipping')
    else:
        log_line(f'[data] MANIFEST: train={len(train_imgs)} val={len(val_imgs)}')
    return sorted(set(train_imgs)), sorted(set(val_imgs))


def _index_indoor_recursive(root: str,
                            holdout_subdir: str,
                            valid_exts: tuple) -> Tuple[List[str], List[str]]:
    train_imgs: List[str] = []
    val_imgs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if not _is_valid_image(full, valid_exts):
                continue
            rel_parts = set(os.path.relpath(full, root).split(os.sep))
            if holdout_subdir and holdout_subdir in rel_parts:
                val_imgs.append(full)
            else:
                train_imgs.append(full)
    log_line(f'[data] RECURSIVE: train={len(train_imgs)} val={len(val_imgs)}')
    return sorted(set(train_imgs)), sorted(set(val_imgs))


def index_indoor_dataset(root: str, holdout_subdir: str,
                         valid_exts: tuple) -> Tuple[List[str], List[str]]:
    train_imgs: List[str] = []
    val_imgs:   List[str] = []

    if not os.path.exists(root):
        log_line(f'[data] WARN: Indoor dataset root not found: {root}')
        return train_imgs, val_imgs

    log_line(f'[data] Indexing indoor dataset: {root}')
    if os.path.isfile(root):
        if _is_valid_image(root, valid_exts):
            log_line('[data] SINGLE FILE: 1 img')
            return [os.path.abspath(root)], []
        train_imgs, val_imgs = _index_indoor_manifest(root, holdout_subdir, valid_exts)
        log_line(f'[data] indoor total train={len(train_imgs)} val={len(val_imgs)}')
        return train_imgs, val_imgs

    direct_imgs = [
        os.path.join(root, f) for f in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, f)) and _is_valid_image(f, valid_exts)
    ]
    if direct_imgs:
        train_imgs, val_imgs = _index_indoor_recursive(root, holdout_subdir, valid_exts)
        log_line(f'[data] indoor total train={len(train_imgs)} val={len(val_imgs)}')
        return train_imgs, val_imgs

    for subdir in sorted(os.listdir(root)):
        sp = os.path.join(root, subdir)
        if not os.path.isdir(sp):
            continue
        img_dir = os.path.join(sp, 'images') if os.path.isdir(os.path.join(sp, 'images')) else sp
        imgs = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))
                if os.path.splitext(f)[1].lower() in valid_exts]
        if not imgs:
            continue
        if subdir == holdout_subdir:
            val_imgs.extend(imgs)
            log_line(f'[data] HOLDOUT {subdir}: {len(imgs)} imgs')
        else:
            train_imgs.extend(imgs)
            log_line(f'[data] TRAIN {subdir}: {len(imgs)} imgs')

    if not train_imgs and not val_imgs:
        train_imgs, val_imgs = _index_indoor_recursive(root, holdout_subdir, valid_exts)
    log_line(f'[data] indoor total train={len(train_imgs)} val={len(val_imgs)}')
    return train_imgs, val_imgs


def build_indoor_real_items(paths: List[str], kind: str = 'indoor_real') -> List[Dict]:
    return [
        {
            'img': path,
            'mask': None,
            'kind': kind,
            'case_id': os.path.splitext(os.path.basename(path))[0],
            'source': 'indoor_real',
        }
        for path in paths
    ]


def sample_indoor_real_subset(paths: List[str], cap: int, seed: int) -> List[str]:
    if cap is None or cap <= 0 or len(paths) <= cap:
        return list(paths)
    rng = random.Random(int(seed))
    chosen = list(paths)
    rng.shuffle(chosen)
    return sorted(chosen[:cap])


def index_imd2020(root_dir: str,
                  valid_exts: tuple,
                  val_split: float = 0.10,
                  split_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    train_items: List[Dict] = []
    val_items:   List[Dict] = []

    if not os.path.exists(root_dir):
        log_line(f'[data] WARN: IMD2020 root not found: {root_dir}')
        return train_items, val_items

    log_line(f'[data] Indexing IMD2020: {root_dir}')
    subdirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    split_rng = random.Random(int(split_seed))
    shuffled  = list(subdirs)
    split_rng.shuffle(shuffled)
    n_val    = int(len(shuffled) * float(val_split))
    val_dirs = set(shuffled[:n_val])

    n_real = 0
    n_splice = 0
    n_skipped_unmasked = 0

    for subdir in subdirs:
        sp    = os.path.join(root_dir, subdir)
        files = sorted(os.listdir(sp))
        orig_img = next(
            (f for f in files
             if '_orig' in os.path.splitext(f)[0]
             and os.path.splitext(f)[1].lower() in valid_exts),
            None,
        )
        if orig_img is None:
            continue

        bucket = val_items if subdir in val_dirs else train_items
        bucket.append({
            'img': os.path.join(sp, orig_img),
            'mask': None,
            'kind': 'imd_real',
            'case_id': subdir,
            'source': 'imd2020',
        })
        n_real += 1

        masks = {
            f.replace('_mask.png', ''): os.path.join(sp, f)
            for f in files if f.endswith('_mask.png')
        }
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_exts:
                continue
            if f == orig_img or f.endswith('_mask.png'):
                continue
            base = os.path.splitext(f)[0]
            mask_path = masks.get(base)
            if not mask_path:
                n_skipped_unmasked += 1
                continue
            bucket.append({
                'img': os.path.join(sp, f),
                'mask': mask_path,
                'kind': 'imd_splice',
                'case_id': subdir,
                'source': 'imd2020',
            })
            n_splice += 1

    log_line(
        f'[data] IMD2020: train={len(train_items)} val={len(val_items)} '
        f'real_cases={n_real} splice_items={n_splice} '
        f'skipped_unmasked={n_skipped_unmasked}'
    )
    return train_items, val_items


def _parse_casia_base_ids(base_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not base_name.startswith('casia_'):
        return None, None
    stem  = base_name[len('casia_'):]
    parts = stem.split('_', 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def index_casia_exported(root_dir: str,
                         valid_exts: tuple,
                         val_split: float = 0.15,
                         split_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    train_items: List[Dict] = []
    val_items:   List[Dict] = []

    if not os.path.exists(root_dir):
        log_line(f'[data] WARN: CASIA exported root not found: {root_dir}')
        return train_items, val_items

    img_dir  = os.path.join(root_dir, 'images')
    mask_dir = os.path.join(root_dir, 'masks')
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        log_line(f'[data] WARN: CASIA exported format requires images/ and masks/: {root_dir}')
        return train_items, val_items

    log_line(f'[data] Indexing CASIA exported: {root_dir}')
    reals: Dict[str, str] = {}
    fakes: Dict[str, str] = {}
    masks: Dict[str, str] = {}

    for name in sorted(os.listdir(img_dir)):
        full = os.path.join(img_dir, name)
        if not os.path.isfile(full):
            continue
        ext  = os.path.splitext(name)[1].lower()
        if ext not in valid_exts:
            continue
        stem = os.path.splitext(name)[0]
        if stem.endswith('_real'):
            reals[stem[:-5]] = full
        elif stem.endswith('_fake'):
            fakes[stem[:-5]] = full

    for name in sorted(os.listdir(mask_dir)):
        full = os.path.join(mask_dir, name)
        if not os.path.isfile(full):
            continue
        if not name.lower().endswith('.png'):
            continue
        stem = os.path.splitext(name)[0]
        if stem.endswith('_mask'):
            masks[stem[:-5]] = full

    pair_bases = sorted(set(reals) & set(fakes) & set(masks))
    if not pair_bases:
        log_line('[data] CASIA: no complete {real, fake, mask} triplets found')
        return train_items, val_items

    pairs: List[Dict] = []
    malformed = 0
    for base in pair_bases:
        bg_id, fg_id = _parse_casia_base_ids(base)
        if not bg_id or not fg_id:
            malformed += 1
            continue
        pairs.append({
            'base': base,
            'real_path': reals[base],
            'fake_path': fakes[base],
            'mask_path': masks[base],
            'bg_id': bg_id,
            'fg_id': fg_id,
        })

    all_bgs = sorted(set(p['bg_id'] for p in pairs))
    all_fgs = sorted(set(p['fg_id'] for p in pairs))
    split_rng = random.Random(int(split_seed))
    split_rng.shuffle(all_bgs)
    split_rng.shuffle(all_fgs)
    n_val_bgs  = int(len(all_bgs) * float(val_split))
    n_val_fgs  = int(len(all_fgs) * float(val_split))
    val_bgs    = set(all_bgs[:n_val_bgs])
    train_bgs  = set(all_bgs[n_val_bgs:])
    val_fgs    = set(all_fgs[:n_val_fgs])
    train_fgs  = set(all_fgs[n_val_fgs:])

    kept_pairs = 0
    discarded  = 0
    for pair in pairs:
        bg_id = pair['bg_id']
        fg_id = pair['fg_id']
        if bg_id in train_bgs and fg_id in train_fgs:
            bucket = train_items
        elif bg_id in val_bgs and fg_id in val_fgs:
            bucket = val_items
        else:
            discarded += 1
            continue
        bucket.append({
            'img': pair['real_path'],
            'mask': None,
            'kind': 'imd_real',
            'case_id': pair['base'],
            'source': 'casia',
            'bg_id': bg_id,
            'fg_id': fg_id,
        })
        bucket.append({
            'img': pair['fake_path'],
            'mask': pair['mask_path'],
            'kind': 'imd_splice',
            'case_id': pair['base'],
            'source': 'casia',
            'bg_id': bg_id,
            'fg_id': fg_id,
        })
        kept_pairs += 1

    log_line(
        f'[data] CASIA: train={len(train_items)} val={len(val_items)} '
        f'triplets_kept={kept_pairs} discarded_mismatched={discarded} malformed={malformed}'
    )
    return train_items, val_items


def _inpaint_clean_name(filename: str) -> str:
    """Strip extension and common modified/original/mask suffixes for matching."""
    stem = os.path.splitext(filename)[0]
    for suf in ('_modified', '_original', '_orig', '_mask', '_fake', '_real',
                '_inpainted', '_gt'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def index_inpaint_triplet(
    root_dir: str,
    valid_exts: tuple,
    *,
    source: str,
    modified_subdir: str = 'modified',
    original_subdir: str = 'original',
    mask_subdir: str = 'mask',
    val_split: float = 0.10,
    split_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Index an inpainting dataset laid out as three parallel folders.

    Layout (v6 COCO-inpaint / SAGID convention)::

        root_dir/<modified_subdir>/   inpainted fakes
        root_dir/<original_subdir>/   pristine originals (pre-inpaint)
        root_dir/<mask_subdir>/       inpaint-region masks

    Files are matched across folders by a cleaned basename (extension and
    modified/original/mask suffixes stripped).  Each matched triplet yields:

      - a real negative item   (kind='imd_real',   mask=None)
      - a fake positive item   (kind='casia_splice', mask=<mask>,
                                real_path=<original>)

    The fake carries ``real_path`` so LabDataset can paste the pristine
    background over the un-edited region — making the SD-inpaint behave like a
    true splice instead of a whole-image VAE fingerprint.  This mirrors v6's
    ``needs_paste=True`` policy for the SD-inpaint family.

    Train/val split is by cleaned base id (a real+fake+mask triplet never
    straddles the split).

    Returns (train_items, val_items).
    """
    train_items: List[Dict] = []
    val_items:   List[Dict] = []

    if not root_dir or not os.path.isdir(root_dir):
        log_line(f'[data] WARN: inpaint root not found ({source}): {root_dir!r}')
        return train_items, val_items

    mod_dir  = os.path.join(root_dir, modified_subdir)
    orig_dir = os.path.join(root_dir, original_subdir)
    mask_dir = os.path.join(root_dir, mask_subdir)
    for label, d in (('modified', mod_dir), ('original', orig_dir), ('mask', mask_dir)):
        if not os.path.isdir(d):
            log_line(f'[data] WARN: inpaint {source} missing {label} dir: {d!r}')
            return train_items, val_items

    def _index_dir(d: str, exts: tuple) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            if not os.path.isfile(full):
                continue
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            out[_inpaint_clean_name(name)] = full
        return out

    mods   = _index_dir(mod_dir, valid_exts)
    origs  = _index_dir(orig_dir, valid_exts)
    # masks are usually PNG; accept valid_exts plus .png regardless
    mask_exts = tuple(set(valid_exts) | {'.png'})
    masks  = _index_dir(mask_dir, mask_exts)

    bases = sorted(set(mods) & set(origs) & set(masks))
    n_missing = len(mods) - len(bases)
    if not bases:
        log_line(
            f'[data] inpaint {source}: no complete (modified, original, mask) '
            f'triplets in {root_dir!r} '
            f'(mods={len(mods)} origs={len(origs)} masks={len(masks)})'
        )
        return train_items, val_items

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    n_val     = int(len(shuffled) * float(val_split))
    val_bases = set(shuffled[:n_val])

    for base in bases:
        bucket = val_items if base in val_bases else train_items
        bucket.append({
            'img':   origs[base],
            'mask':  None,
            'kind':  'imd_real',
            'case_id': f'{source}_{base}',
            'source': source,
        })
        bucket.append({
            'img':   mods[base],
            'mask':  masks[base],
            'kind':  'casia_splice',          # treated identically to a splice
            'case_id': f'{source}_{base}',
            'source': source,
            'real_path': origs[base],         # drives pristine-background paste
        })

    log_line(
        f'[data] inpaint {source}: train={len(train_items)} val={len(val_items)} '
        f'triplets={len(bases)} unmatched_mods={n_missing}'
    )
    return train_items, val_items


# ── BFree (COCO-anchored SD2.1 inpainting; diffcat=bbox-mask, samecat=exact) ──

def _normalize_dir_name(name: str) -> str:
    base = name.lower().rstrip('/\\')
    return base[:-4] if base.endswith('.zip') else base


def _resolve_named_subdir(root_dir: Optional[str], desired_names) -> Optional[str]:
    """Case-insensitive lookup of a subdir under root_dir by any desired name."""
    if not root_dir or not os.path.isdir(root_dir):
        return None
    desired = {_normalize_dir_name(n) for n in desired_names}
    for entry in sorted(os.listdir(root_dir)):
        ep = os.path.join(root_dir, entry)
        if os.path.isdir(ep) and _normalize_dir_name(entry) in desired:
            return ep
    return None


def _resolve_bfree_root(root_dir: str) -> str:
    """Accept either the BFree dataset root itself or a parent containing it."""
    if not root_dir or not os.path.isdir(root_dir):
        return root_dir
    required = {'coco_real_512', 'sd2.1_inpainted_diffcat',
                'sd2.1_inpainted_samecat', 'bbox'}
    mask_names = {'masks', 'mask'}

    def _entries(d):
        return {_normalize_dir_name(e) for e in os.listdir(d)
                if os.path.isdir(os.path.join(d, e))}

    if required.issubset(_entries(root_dir)) and _entries(root_dir) & mask_names:
        return root_dir
    for entry in sorted(os.listdir(root_dir)):
        ep = os.path.join(root_dir, entry)
        if os.path.isdir(ep):
            sub = _entries(ep)
            if required.issubset(sub) and sub & mask_names:
                return ep
    return root_dir


def _bfree_file_dict(folder: Optional[str], valid_exts: tuple) -> Dict[str, str]:
    """Recursively map cleaned (lowercased, suffix-stripped) basename → path."""
    out: Dict[str, str] = {}
    if not folder or not os.path.isdir(folder):
        return out
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in valid_exts:
                continue
            out[_inpaint_clean_name(f).lower()] = os.path.join(root, f)
    return out


def index_bfree(
    root_dir: str,
    valid_exts: tuple,
    *,
    source: str = 'bfree',
    val_split: float = 0.10,
    split_seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Index the BFree COCO-anchored SD2.1 inpainting dataset.

    Layout (under ``root_dir``, or a parent that contains it)::

        COCO_real_512/              anchor real images
        SD2.1_inpainted_diffcat/    inpainted fakes (different category)
        SD2.1_inpainted_samecat/    inpainted fakes (same category)
        masks/ (or mask/)           exact segmentation masks
        bbox/                       bounding-box masks

    Each COCO id yields one anchor real plus up to two fakes (diffcat, samecat).
    Operative-mask policy mirrors v6: ``diffcat`` uses the bbox mask (falling
    back to exact), ``samecat`` uses the exact segmentation mask. Fakes carry
    ``real_path`` so LabDataset pastes the pristine background over the un-edited
    region — true-splice behavior, not a whole-image VAE fingerprint.

    Items follow the inpaint-triplet schema (real: kind='imd_real';
    fake: kind='casia_splice'). Train/val split is by group id so a real never
    straddles its fakes. Returns (train_items, val_items).
    """
    train_items: List[Dict] = []
    val_items:   List[Dict] = []

    if not root_dir or not os.path.isdir(root_dir):
        log_line(f'[data] WARN: bfree root not found: {root_dir!r}')
        return train_items, val_items
    root_dir = _resolve_bfree_root(root_dir)

    anchor_dir = _resolve_named_subdir(root_dir, ['COCO_real_512'])
    mask_dir   = _resolve_named_subdir(root_dir, ['masks', 'mask'])
    bbox_dir   = _resolve_named_subdir(root_dir, ['bbox'])
    target_specs = [
        ('diffcat', _resolve_named_subdir(root_dir, ['SD2.1_inpainted_diffcat'])),
        ('samecat', _resolve_named_subdir(root_dir, ['SD2.1_inpainted_samecat'])),
    ]
    if anchor_dir is None:
        log_line(f'[data] WARN: bfree anchor dir (COCO_real_512) not found '
                 f'under {root_dir!r}')
        return train_items, val_items

    mask_exts = tuple(set(valid_exts) | {'.png'})
    anchor_d  = _bfree_file_dict(anchor_dir, valid_exts)
    mask_d    = _bfree_file_dict(mask_dir, mask_exts)
    bbox_d    = _bfree_file_dict(bbox_dir, mask_exts)

    # Collect fakes first so the train/val split can be by group id.
    fakes: List[Tuple[str, str, str, Optional[str]]] = []  # base, variant, path, mask
    for variant, target_dir in target_specs:
        if target_dir is None:
            log_line(f'[data] bfree: missing target dir for {variant}')
            continue
        for base, fake_path in _bfree_file_dict(target_dir, valid_exts).items():
            if base not in anchor_d:
                continue
            exact = mask_d.get(base)
            bbox  = bbox_d.get(base) if variant == 'diffcat' else None
            fakes.append((base, variant, fake_path, bbox or exact))

    bases = sorted({b for b, _, _, _ in fakes})
    if not bases:
        log_line(f'[data] bfree: no matched (anchor, fake) pairs under {root_dir!r} '
                 f'(anchors={len(anchor_d)}, masks={len(mask_d)}, bbox={len(bbox_d)})')
        return train_items, val_items

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    val_bases = set(shuffled[: int(len(shuffled) * float(val_split))])

    seen_real = set()
    for base, variant, fake_path, mask in fakes:
        bucket = val_items if base in val_bases else train_items
        if base not in seen_real:
            bucket.append({
                'img':   anchor_d[base],
                'mask':  None,
                'kind':  'imd_real',
                'case_id': f'{source}_{base}',
                'source': source,
            })
            seen_real.add(base)
        bucket.append({
            'img':   fake_path,
            'mask':  mask,
            'kind':  'casia_splice',          # treated identically to a splice
            'case_id': f'{source}_{base}_{variant}',
            'source': source,
            'variant': variant,
            'real_path': anchor_d[base],       # drives pristine-background paste
        })

    n_masked = sum(1 for _, _, _, m in fakes if m is not None)
    log_line(
        f'[data] bfree: train={len(train_items)} val={len(val_items)} '
        f'bases={len(bases)} fakes={len(fakes)} masked={n_masked} '
        f'anchors={len(anchor_d)}'
    )
    if n_masked < len(fakes):
        log_line(f'[data] bfree WARN: {len(fakes) - n_masked} fakes missing an '
                 f'operative mask (will fall back to whole-image).')
    return train_items, val_items
