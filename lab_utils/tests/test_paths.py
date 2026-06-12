from types import SimpleNamespace

from lab_utils.paths import DataPaths, resolve_data_paths


def test_resolve_data_paths_prefers_args_over_env(monkeypatch):
    monkeypatch.setenv("DINO_IMD2020_ROOT", "/env/imd")
    args = SimpleNamespace(imd2020_root="/arg/imd", casia_root="/arg/casia")
    paths = resolve_data_paths(args)
    assert paths.imd2020_root == "/arg/imd"
    assert paths.casia_root == "/arg/casia"


def test_resolve_data_paths_uses_env(monkeypatch):
    monkeypatch.setenv("DINO_CASIA_ROOT", "/env/casia")
    monkeypatch.setenv("DINO_RUN_ROOT", "/env/runs")
    paths = resolve_data_paths()
    assert paths == DataPaths(casia_root="/env/casia", run_root="/env/runs")


def test_resolve_data_paths_explicit_overrides_everything(monkeypatch):
    monkeypatch.setenv("DINO_RUN_ROOT", "/env/runs")
    paths = resolve_data_paths(run_root="/explicit/runs")
    assert paths.run_root == "/explicit/runs"
