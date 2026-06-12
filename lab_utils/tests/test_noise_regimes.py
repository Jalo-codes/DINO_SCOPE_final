import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from PIL import Image, ImageDraw

from lab_utils.data.dataset import LabDataset, lab_collate_fn
from lab_utils.data.augment.corruptions import CorruptionSpec
from lab_utils.data.resolution import Resolution


def _write_pair(tmp):
    img = Image.new('RGB', (80, 80), (120, 140, 160))
    mask = Image.new('L', (80, 80), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle((20, 20, 55, 55), fill=255)
    img_path = os.path.join(tmp, 'img.png')
    mask_path = os.path.join(tmp, 'mask.png')
    img.save(img_path)
    mask.save(mask_path)
    return img_path, mask_path


def test_splice_mask_corruption_sets_weight_and_noise_meta():
    with tempfile.TemporaryDirectory() as tmp:
        img_path, mask_path = _write_pair(tmp)
        ds = LabDataset(
            [{'img': img_path, 'mask': mask_path, 'kind': 'imd_splice'}],
            Resolution(64, 16),
            augment=True,
            use_splice_degradation=True,
            splice_mask_corrupt_prob=1.0,
            splice_mask_loss_weight=0.2,
            light_aug_kwargs={
                'jpeg_prob': 0.0,
                'noise_prob': 0.0,
                'resize_prob': 0.0,
                'flip_prob': 0.0,
            },
            degradation_kwargs={
                'families': ('jpeg',),
                'jpeg_q_min': 35,
                'jpeg_q_max': 35,
            },
        )
        sample = ds[0]
        assert float(sample['splice_loss_weight']) == 0.2
        assert bool(sample['degrade_supervised'])
        assert sample['meta']['noise_region'] == 'splice_mask'
        assert sample['meta']['noise_family'] == 'jpeg'


def test_eval_aug_global_corruption_is_collated_in_meta():
    with tempfile.TemporaryDirectory() as tmp:
        img_path, mask_path = _write_pair(tmp)
        ds = LabDataset(
            [{'img': img_path, 'mask': mask_path, 'kind': 'imd_splice'}],
            Resolution(64, 16),
            augment=False,
            eval_aug_mode='global_gaussian',
            eval_corruption_spec=CorruptionSpec('gaussian', {'std': 0.24}, severity_tag='eval'),
            eval_corruption_region='global',
        )
        batch = lab_collate_fn([ds[0]])
        assert batch['img'].shape == (1, 3, 64, 64)
        assert batch['meta'][0]['noise_family'] == 'gaussian'
        assert batch['meta'][0]['noise_region'] == 'global'
        assert batch['meta'][0]['whole_image_aug'] == 1
