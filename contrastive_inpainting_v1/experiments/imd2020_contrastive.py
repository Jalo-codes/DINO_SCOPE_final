"""contrastive_inpainting_v1.experiments.imd2020_contrastive

IMD2020 + CASIA + indoor-real contrastive experiment.
Parity target for the existing harness_train.py run.
"""

import dataclasses
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from lab_utils.data.indexer import (
    index_imd2020,
    index_casia_exported,
    index_indoor_dataset,
    build_indoor_real_items,
    sample_indoor_real_subset,
)
from lab_utils.logging.text import log_line


def _source_counts(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(Counter(str(i.get('source', 'unknown')) for i in items).items()))


@dataclasses.dataclass
class IMD2020ContrastiveSpec:
    """Build item lists for IMD2020 + CASIA + indoor-real contrastive run.

    Args:
        imd2020_root:    Path to IMD2020 directory (None = skip).
        casia_root:      Path to CASIA exported directory (None = skip).
        casia_train:     If false, all indexed CASIA items are validation-only.
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
    indoor_holdout:   str  = 'unclassified'
    imd_val_split:    float = 0.10
    imd_split_seed:   int   = 42
    casia_val_split:  float = 0.15
    casia_split_seed: int   = 42
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
            train_items.extend(t)
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
            f'[data] imd2020_contrastive: '
            f'train={len(train_items)} val={len(val_items)} '
            f'casia_train={int(self.casia_train)}'
        )
        log_line(
            f'[data] imd2020_contrastive sources: '
            f'train={_source_counts(train_items)} val={_source_counts(val_items)}'
        )
        return train_items, val_items
