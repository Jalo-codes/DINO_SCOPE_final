import importlib


def test_v3_core_imports():
    for name in [
        "contrastive_inpainting_v1.configs.base",
        "contrastive_inpainting_v1.experiments.imd2020_bce",
        "contrastive_inpainting_v1.pipeline.cli",
    ]:
        importlib.import_module(name)
