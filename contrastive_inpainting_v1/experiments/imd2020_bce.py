"""contrastive_inpainting_v1.experiments.imd2020_bce

IMD2020 + CASIA + indoor-real BCE experiment.
Item structure is identical to imd2020_contrastive.  The difference is in
the training hooks (selective_bce_loss instead of selective_contrastive_loss),
which live in the training script, not here.
"""

import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from lab_utils.data.indexer import (
    index_imd2020,
    index_casia_exported,
    index_indoor_dataset,
    index_inpaint_triplet,
    index_anyedit,
    index_bfree,
    build_indoor_real_items,
    sample_indoor_real_subset,
)
from lab_utils.logging.text import log_line


def _source_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(i.get('source', 'unknown')) for i in items).items()))


@dataclasses.dataclass
class IMD2020BCESpec:
    """Build item lists for IMD2020 + CASIA + indoor-real BCE run.

    Args:
        imd2020_root:    Path to IMD2020 directory (None = skip).
        casia_root:      Path to CASIA exported directory (None = skip).
        imd_train:       If False, all indexed IMD2020 items (both the train and
                         val halves of the indexer split) are routed to the
                         validation set — i.e. IMD2020 is held out entirely as
                         an OOD generalization check. Pair with casia_train=True
                         for the train-on-CASIA / val-on-IMD flip.
        casia_train:     If False, all indexed CASIA items are validation-only.
        indoor_root:     Path to indoor real images (None = skip).
        indoor_holdout:  Holdout subdirectory name for indoor real.
        imd_val_split:   Val fraction for IMD2020.
        imd_split_seed:  Seed for IMD2020 split.
        casia_val_split: Val fraction for CASIA.
        casia_split_seed: Seed for CASIA split.
        indoor_real_cap: Max indoor real items in train set.
    """
    imd2020_root:     Optional[str] = None
    casia_root:       Optional[str] = None
    indoor_root:      Optional[str] = None
    # Phase-2 inpainting sources (SD-inpaint family, pasted to behave as splices).
    # COCO-inpaint:  <root>/images/{modified,original,mask}
    # SAGID:         <root>/{modified,original,mask}
    coco_inpaint_root: Optional[str] = None
    sagid_root:        Optional[str] = None
    # BFree: <root>/{COCO_real_512, SD2.1_inpainted_diffcat,
    #                SD2.1_inpainted_samecat, masks|mask, bbox}
    bfree_root:        Optional[str] = None
    # AnyEdit: <root>/images/{*_real.*,*_fake.*}  <root>/masks/
    anyedit_root:      Optional[str] = None
    indoor_holdout:   str  = 'unclassified'
    imd_val_split:    float = 0.10
    imd_split_seed:   int   = 42
    casia_val_split:  float = 0.15
    casia_split_seed: int   = 42
    inpaint_val_split: float = 0.10
    inpaint_split_seed: int  = 42
    imd_train:        bool  = True
    casia_train:      bool  = False
    indoor_real_cap:  int   = 512

    def build_items(self, cfg) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        exts = tuple(cfg.valid_exts)
        train_items: List[Dict] = []
        val_items:   List[Dict] = []

        if self.imd2020_root:
            t, v = index_imd2020(
                self.imd2020_root, exts,
                val_split=self.imd_val_split,
                split_seed=self.imd_split_seed,
            )
            if self.imd_train:
                train_items.extend(t)
                val_items.extend(v)
            else:
                # IMD held out entirely as OOD validation (train-on-CASIA flip).
                val_items.extend(t)
                val_items.extend(v)

        if self.casia_root:
            t, v = index_casia_exported(
                self.casia_root, exts,
                val_split=self.casia_val_split,
                split_seed=self.casia_split_seed,
            )
            if self.casia_train:
                train_items.extend(t)
                val_items.extend(v)
            else:
                val_items.extend(t)
                val_items.extend(v)

        # ── Inpainting sources (always join the TRAIN mix; held-out val slice
        #    used for in-training localization eval). TGIF is the OOD hold and
        #    is handled separately by the eval path, not here. ──
        if self.coco_inpaint_root:
            t, v = index_inpaint_triplet(
                self.coco_inpaint_root, exts, source='coco_inpaint',
                modified_subdir='images/modified',
                original_subdir='images/original',
                mask_subdir='images/mask',
                val_split=self.inpaint_val_split,
                split_seed=self.inpaint_split_seed,
            )
            train_items.extend(t)
            val_items.extend(v)

        if self.sagid_root:
            t, v = index_inpaint_triplet(
                self.sagid_root, exts, source='sagid',
                modified_subdir='modified',
                original_subdir='original',
                mask_subdir='mask',
                val_split=self.inpaint_val_split,
                split_seed=self.inpaint_split_seed,
            )
            train_items.extend(t)
            val_items.extend(v)

        if self.bfree_root:
            t, v = index_bfree(
                self.bfree_root, exts, source='bfree',
                val_split=self.inpaint_val_split,
                split_seed=self.inpaint_split_seed,
            )
            train_items.extend(t)
            val_items.extend(v)

        if self.anyedit_root:
            t, v = index_anyedit(
                self.anyedit_root, exts, source='anyedit',
                val_split=self.inpaint_val_split,
                split_seed=self.inpaint_split_seed,
            )
            train_items.extend(t)
            val_items.extend(v)

        if self.indoor_root:
            train_paths, val_paths = index_indoor_dataset(
                self.indoor_root, self.indoor_holdout, exts
            )
            train_paths = sample_indoor_real_subset(
                train_paths, self.indoor_real_cap, seed=self.imd_split_seed
            )
            train_items.extend(build_indoor_real_items(train_paths))
            val_items.extend(build_indoor_real_items(val_paths, kind='indoor_real'))

        log_line(
            f'[data] imd2020_bce: '
            f'train={len(train_items)} val={len(val_items)} '
            f'imd_train={int(self.imd_train)} casia_train={int(self.casia_train)}'
        )
        log_line(
            f'[data] imd2020_bce sources: '
            f'train={_source_counts(train_items)} val={_source_counts(val_items)}'
        )
        return train_items, val_items
