from pathlib import Path


FORBIDDEN = (
    "BucketSizeDetector",
    "lab_utils.model.losses.buckets",
    "lab_utils.eval.buckets",
    "bucket_size_spec",
    "train_bucket_size",
    "eval_bucket_size",
    "cost_matrices",
)


def test_v3_does_not_import_bucket_prediction_modules():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        if path == Path(__file__):
            continue
        text = path.read_text()
        for token in FORBIDDEN:
            if token in text:
                offenders.append((str(path.relative_to(root)), token))
    assert offenders == []
