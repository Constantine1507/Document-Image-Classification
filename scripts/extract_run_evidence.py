"""Extract recorded metrics and figures from the unmodified Kaggle notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/executed/v2_gpu_fix_kaggle.ipynb"
EXPECTED_SHA256 = "1af2fc26eabea894bfec9dbb18884ee8678b510ace3ee783927cfd34fdccb758"


def output_text(output):
    """Read text output without replacing it with inferred experiment results."""
    return "".join(output.get("text", []) or output.get("data", {}).get("text/plain", []))


def json_objects(text):
    """Find complete JSON objects mixed into notebook console output."""
    decoder, position, objects = json.JSONDecoder(), 0, []
    while (start := text.find("{", position)) >= 0:
        try:
            value, length = decoder.raw_decode(text[start:])
            objects.append(value)
            position = start + length
        except json.JSONDecodeError:
            position = start + 1
    return objects


def save_json(path, value):
    """Write an extracted record using strict JSON without fabricated NaNs."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def html_table(output):
    """Recover the displayed DataFrame, retaining its displayed precision."""
    frame = pd.read_html(io.StringIO("".join(output["data"]["text/html"])))[0]
    first = frame.columns[0]
    if str(first).startswith("Unnamed:") and frame[first].tolist() == list(range(len(frame))):
        frame = frame.drop(columns=[first])
    return frame


def main():
    """Export history, validation metrics, audit, figures, and provenance."""
    payload = NOTEBOOK.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError("Executed notebook changed; verify the evidence before re-extracting")
    notebook = json.loads(payload)
    if any(o.get("output_type") == "error" for c in notebook["cells"] for o in c.get("outputs", [])):
        raise ValueError("Executed notebook contains errors")
    cells = notebook["cells"]
    output = ROOT / "results/v2_gpu_fix"
    output.mkdir(parents=True, exist_ok=True)
    audit = next(v for v in json_objects(output_text(cells[18]["outputs"][0])) if "fingerprint" in v)
    save_json(output / "data_audit.json", audit)
    html_table(cells[18]["outputs"][1]).to_csv(output / "split_class_counts.csv", index=False)
    best_epochs = {}
    for name, train_index, summary_index, eval_index in [
        ("convnext_tiny", 2, 4, 0), ("efficientnet_v2_s", 6, 8, 4)
    ]:
        target = output / name
        target.mkdir(exist_ok=True)
        history = [v for v in json_objects(output_text(cells[22]["outputs"][train_index])) if "epoch" in v]
        if len(history) != 20:
            raise ValueError(f"Expected 20 recorded epochs for {name}, found {len(history)}")
        pd.DataFrame(history).to_csv(target / "history.csv", index=False)
        summary = next(v for v in json_objects(output_text(cells[22]["outputs"][summary_index])) if "model" in v)
        metrics = next(v for v in json_objects(output_text(cells[25]["outputs"][eval_index])) if "accuracy" in v)
        save_json(target / "summary.json", summary)
        save_json(target / "metrics.json", metrics)
        best_epochs[name] = sorted(history, key=lambda v: (-v["val_accuracy"], v["val_log_loss"]))[0]["epoch"]
        html_table(cells[25]["outputs"][eval_index + 1]).rename(
            columns={"Unnamed: 0": "class"}).to_csv(target / "classification_report.csv", index=False)
        for offset, filename in [(2, "training_curves.png"), (3, "confusion_matrix.png")]:
            encoded = cells[25]["outputs"][eval_index + offset]["data"]["image/png"]
            (target / filename).write_bytes(base64.b64decode(encoded))
    comparison = html_table(cells[27]["outputs"][1])
    comparison[["model", "accuracy", "macro_f1", "log_loss"]].to_csv(output / "model_comparison.csv", index=False)
    save_json(output / "kaggle_score_reported.json", {
        "model": "ensemble_equal", "score": 0.88456,
        "source": "reported by repository owner in the project conversation",
        "independently_verified": False, "leaderboard_split": None,
        "rank": None, "submission_sha256": None,
    })
    save_json(output / "evidence_manifest.json", {
        "notebook": NOTEBOOK.relative_to(ROOT).as_posix(), "sha256": actual_hash,
        "recorded_epochs_per_model": 20, "best_ema_epochs": best_epochs,
        "selected_eval_sizes": {"convnext_tiny": [384], "efficientnet_v2_s": [384]},
        "selected_eval_sizes_evidence": "Only 384px test inference appears in cell 25 logs",
        "comparison_precision": "Six decimals displayed by the executed notebook",
        "submission_shape_recorded": [7994, 2],
        "limitations": ["No checkpoints or complete submission CSV supplied",
                        "No raw prediction probabilities supplied",
                        "Clean-train diagnostic file was not printed in the notebook",
                        "Notebook metadata predates its recorded completed training output",
                        "This archive predates the maintained v2.2 source; its outputs were not rerun"],
        "gpu_diagnostics_log": output_text(cells[21]["outputs"][0]),
    })
    print(json.dumps({"status": "extracted", "epochs": 40, "best_ema_epochs": best_epochs,
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
