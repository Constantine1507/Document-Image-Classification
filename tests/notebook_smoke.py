"""Execute every code cell on synthetic TIFF data using both actual backbones."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
import nbformat
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from smoke_test import make_fixture


def main():
    """Run the notebook in order with tiny CPU overrides and verify its outputs."""
    notebook = nbformat.read(ROOT / "notebooks/Document_Classification_v2_2_Kaggle.ipynb", as_version=4)
    report_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    original_directory = Path.cwd()
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="docvision-notebook-") as temporary:
            base = Path(temporary)
            dataset = make_fixture(base / "dataset")
            project = base / "project"
            shell = InteractiveShell.instance()
            executed = []
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source
                if "user-config" in cell.metadata.get("tags", []):
                    source += f"\nPROJECT = Path({str(project)!r})\nDATA_ROOT = {str(dataset)!r}\n"
                    source += "\n".join([
                        "REQUIRE_GPU = False", "EXPECTED_CLASSES = 3", "AUDIT_WORKERS = 2",
                        "REUSE_LEGACY_SPLIT = False", "RUN_TAG = 'synthetic_cpu_validation'",
                        "COMMON_OVERRIDES = dict(pretrained=False, epochs=1, finetune_epoch=1, "
                        "min_epochs=1, patience=2, image_size=64, finetune_size=64, val_size=64, "
                        "eval_sizes=[64,96], batch_size=4, eval_batch_size=4, accumulation_steps=2, "
                        "num_workers=0, warmup_epochs=0, clean_train_eval_rows=8)",
                    ])
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    result = shell.run_cell(source, store_history=False)
                if not result.success:
                    raise RuntimeError(f"Cell {index} failed:\n{buffer.getvalue()}") from (result.error_before_exec or result.error_in_exec)
                executed.append(index)
                print(f"Notebook code cell {index}: passed", flush=True)
            submission = pd.read_csv(project / "submission.csv", dtype={"id": str})
            sample = pd.read_csv(dataset / "sample_submission.csv", dtype={"id": str})
            assert list(submission.id) == list(sample.id)
            assert list(submission.columns) == list(sample.columns)
            assert len(submission) == 7
            assert (project / "github_results.zip").is_file()
            assert (project / "results/latest/model_comparison.csv").is_file()
            report = {"status": "passed", "total_cells": len(notebook.cells),
                      "executed_code_cells": len(executed), "cell_indices": executed,
                      "models": ["convnext_tiny", "efficientnet_v2_s"],
                      "data": "synthetic TIFF fixture", "pretrained": False, "device": "cpu",
                      "epochs_per_model": 1, "train_size": 64, "eval_sizes": [64, 96],
                      "submission_rows": len(submission), "competition_accuracy": None,
                      "seconds": time.perf_counter() - started}
    finally:
        os.chdir(original_directory)
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
