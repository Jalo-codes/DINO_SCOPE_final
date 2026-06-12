import argparse

from contrastive_inpainting_v1.pipeline.cli import add_data_path_args


def test_checkpoint_root_alias_maps_to_run_root():
    p = add_data_path_args(argparse.ArgumentParser())
    args = p.parse_args(["--checkpoint_root", "/tmp/runs"])
    assert args.run_root == "/tmp/runs"
