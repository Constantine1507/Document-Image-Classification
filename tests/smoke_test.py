"""Functional tests with synthetic TIFFs; no competition accuracy claim."""
import json
import shutil
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

import docvision_v2.data as data_module
from docvision_v2.data import (ImageResolver, build_submission, load_prepared,
                              open_document, prepare_data)
from docvision_v2.engine import train_epoch, train_model
from docvision_v2.evaluate import compare_models, evaluate_model
from docvision_v2.models import ModelEMA, build_model, build_optimizer, default_config
from docvision_v2.utils import sha256_file, write_json


def make_fixture(root, classes=3):
    """Create small TIFF data with deliberate corruption, conflict and overlap."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    train, test = [], []
    for label in range(classes):
        for index in range(10):
            filename = f"train/class_{label}/{label}_{index}.tif"
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            image = rng.integers(0, 30, (80, 64, 3), dtype=np.uint8)
            image[:, :, label % 3] += 180
            Image.fromarray(image).save(path)
            train.append({"id": f"{label*10+index:05d}", "filename": filename,
                          "label": label, "class_name": f"class_{label}"})
    for index in range(6):
        filename = f"test/test_{index}.tif"
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (70, 60, 3), dtype=np.uint8)).save(path)
        test.append({"id": f"test_{index:05d}", "filename": filename})
    # Same-label duplicate, conflicting-label duplicate, corrupt train image.
    shutil.copy2(root / train[1]["filename"], root / "train/duplicate.tif")
    train.append({"id": "duplicate", "filename": "train/duplicate.tif", "label": 0, "class_name": "class_0"})
    shutil.copy2(root / train[0]["filename"], root / "train/conflict.tif")
    train.append({"id": "conflict", "filename": "train/conflict.tif", "label": 1, "class_name": "class_1"})
    (root / "train/corrupt.tif").write_bytes(b"not an image")
    train.append({"id": "corrupt", "filename": "train/corrupt.tif", "label": 0, "class_name": "class_0"})
    shutil.copy2(root / train[2]["filename"], root / "test/overlap.tif")
    test.append({"id": "overlap_test", "filename": "test/overlap.tif"})
    pd.DataFrame(train).to_csv(root / "train.csv", index=False)
    pd.DataFrame(test).to_csv(root / "test.csv", index=False)
    # Deliberately reverse template order and preserve leading zeros in IDs.
    pd.DataFrame({"id": [row["id"] for row in test[::-1]], "label": 0}).to_csv(root / "sample_submission.csv", index=False)
    return root


def smoke_config():
    """Return a short CPU-only configuration for meaningful pipeline tests."""
    cfg = default_config("smoke_cnn")
    cfg.update(num_classes=3, pretrained=False, require_gpu=False, epochs=2,
               finetune_epoch=2, image_size=32, finetune_size=40, val_size=40,
               eval_sizes=[40, 48], batch_size=4, eval_batch_size=6,
               accumulation_steps=3, num_workers=0, warmup_epochs=1,
               min_epochs=2, patience=3, clean_train_eval_rows=8)
    return cfg


def test_accumulation():
    """Compare a short final microbatch against one exact full-batch update."""
    torch.manual_seed(19)
    x, y = torch.randn(11, 3, 8, 8), torch.randint(0, 3, (11,))
    model = build_model(smoke_config())
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    for group in optimizer.param_groups:
        group["base_lr"] = .01
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    reference_optimizer.zero_grad()
    torch.nn.functional.cross_entropy(reference(x), y).backward()
    reference_optimizer.step()
    cfg = smoke_config()
    cfg.update(accumulation_steps=3, mixup_alpha=0, label_smoothing=0,
               finetune_label_smoothing=0, grad_clip=100, min_lr_ratio=1.)
    train_epoch(model, ModelEMA(model), DataLoader(TensorDataset(x, y, y), batch_size=4),
                optimizer, torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
                cfg, 1, 0, 1, 0)
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(model.parameters(), reference.parameters()))


def test_backbones():
    """Exercise real ConvNeXt Tiny and EfficientNetV2-S forward/backward on CPU."""
    results = []
    for name in ["convnext_tiny", "efficientnet_v2_s"]:
        cfg = default_config(name)
        cfg.update(pretrained=False, num_classes=16, require_gpu=False)
        model = build_model(cfg)
        model.train()
        logits = model(torch.randn(2, 3, 64, 64))
        assert logits.shape == (2, 16)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))
        loss.backward()
        assert model.classifier[-1].weight.grad is not None
        results.append({"model": name, "output_shape": list(logits.shape), "finite_loss": bool(torch.isfinite(loss))})
        del model, logits, loss
    return results


def run_tests(output_report=None):
    """Check data invariants, resume equivalence, ensemble and real backbones."""
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="docvision-v2-test-") as temporary:
        base = Path(temporary)
        root = make_fixture(base / "dataset")
        before = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
        project = base / "project"
        manifest = prepare_data(root, project, workers=2, expected_classes=3)
        assert manifest["conflicting_rows_excluded"] == 2
        assert manifest["corrupt_rows_excluded"] == 1
        assert manifest["train_test_overlap_groups"] == 1
        original_inspector = data_module.inspect_image
        def reject_new_reads(record):
            """Fail if a repeated audit unexpectedly decodes unchanged images."""
            raise AssertionError("Cache reuse unexpectedly decoded an unchanged image")
        data_module.inspect_image = reject_new_reads
        cached = prepare_data(root, project, workers=2, expected_classes=3)
        data_module.inspect_image = original_inspector
        assert cached["fingerprint"] == manifest["fingerprint"]
        after = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
        assert before == after, "Input files were modified"
        train, valid, test, sample, _ = load_prepared(project)
        assert not set(train.pixel_sha256) & set(valid.pixel_sha256)
        assert train.id.str.startswith("0").any(), "Leading zeros were lost"
        # Same paths stored inside an archive: read without extracting images.
        zip_root = base / "zip_dataset"
        zip_root.mkdir()
        for name in ["train.csv", "test.csv", "sample_submission.csv"]:
            shutil.copy2(root / name, zip_root / name)
        with ZipFile(zip_root / "images.zip", "w") as archive:
            for path in root.rglob("*.tif"):
                archive.write(path, "nested/" + path.relative_to(root).as_posix())
        zip_manifest = prepare_data(zip_root, base / "zip_project", workers=2, expected_classes=3)
        assert zip_manifest["fingerprint"] == manifest["fingerprint"]
        assert not list(zip_root.rglob("*.tif")), "ZIP images were extracted"
        image_resolver = ImageResolver([root])
        try:
            image_resolver.resolve("../outside.tif")
            raise AssertionError("Traversal was not rejected")
        except ValueError:
            pass
        # ID-based submission reordering.
        ids = list(test.id)[::-1]
        p = np.eye(3)[np.arange(len(ids)) % 3]
        submission = build_submission(test, sample, ids, p)
        lookup = dict(zip(ids, p.argmax(1)))
        assert submission.label.tolist() == [lookup[value] for value in sample.id]
        cfg = smoke_config()
        run = train_model(project, cfg, "resume", stop_after=1)
        assert json.loads((run / "summary.json").read_text())["completed_epoch"] == 1
        train_model(project, cfg, "resume")
        fresh = train_model(project, cfg, "uninterrupted")
        resumed = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
        uninterrupted = torch.load(fresh / "last.pt", weights_only=True, map_location="cpu")
        assert resumed["epoch"] == 2
        assert all(torch.equal(value, uninterrupted["model_state"][key]) for key, value in resumed["model_state"].items())
        evaluation = evaluate_model(project, run)
        comparison = compare_models(project, [evaluation])
        second_evaluation = base / "second_synthetic_evaluation"
        shutil.copytree(evaluation, second_evaluation)
        second_meta = json.loads((second_evaluation / "metadata.json").read_text())
        second_meta["model"] = "second_synthetic_fixture"
        write_json(second_evaluation / "metadata.json", second_meta)
        comparison = compare_models(project, [evaluation, second_evaluation])
        assert set(comparison.model) == {"smoke_cnn", "second_synthetic_fixture", "ensemble_equal"}
        actual = pd.read_csv(project / "submission.csv", dtype={"id": str})
        assert list(actual.id) == list(sample.id) and len(actual) == len(test)
        # Detect a changed training configuration rather than silently resuming.
        changed = dict(cfg, lr=cfg["lr"] * 2)
        try:
            train_model(project, changed, "resume")
            raise AssertionError("Mismatched resume was not rejected")
        except ValueError:
            pass
        test_accumulation()
        backbones = test_backbones()
        result = {
            "status": "passed", "fixture": "synthetic TIFF images, 3 classes; real data not available",
            "checks": ["direct files without modification/copy", "ZIP streaming without extraction",
                       "cached audit without re-decoding", "corrupt training quarantine", "conflicting-label quarantine",
                       "duplicate-group split isolation", "train/test overlap report", "leading-zero ID preservation",
                       "submission reorder by ID", "checkpoint resume equals uninterrupted CPU run",
                       "reject mismatched resume", "TTA selection/evaluation/submission pipeline",
                       "two-model probability ensemble and selection",
                       "short-tail gradient accumulation equals full batch"],
            "architecture_smoke_tests": backbones,
            "torch": str(torch.__version__), "competition_accuracy": None,
        }
        if output_report:
            write_json(output_report, result)
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    run_tests(sys.argv[1] if len(sys.argv) > 1 else None)
