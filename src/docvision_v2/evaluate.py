"""Evaluate the selected checkpoint, verify TTA, and compare a fixed ensemble."""
import gc
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .data import build_submission, load_prepared, open_document
from .engine import load_inference, metric_dict, predict_loader
from .models import make_loader
from .utils import sha256_file, source_fingerprint, write_json


def report_predictions(frame, probabilities, class_names, output):
    """Save per-class diagnostics and confident mistakes for manual review."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = frame.label.to_numpy(dtype=int)
    prediction = probabilities.argmax(1)
    metrics = metric_dict(labels, probabilities)
    unique = ~frame.pixel_sha256.duplicated().to_numpy()
    metrics["unique_image_group_accuracy"] = float((prediction[unique] == labels[unique]).mean())
    metrics["correct"] = int((prediction == labels).sum())
    metrics["rows"] = len(frame)
    write_json(output / "metrics.json", metrics)
    report = classification_report(labels, prediction, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output / "classification_report.csv")
    rows = frame[["id", "image_path", "label", "pixel_sha256"]].copy()
    rows["prediction"] = prediction
    rows["confidence"] = probabilities.max(1)
    rows["correct"] = prediction == labels
    rows.to_csv(output / "validation_predictions.csv", index=False)
    errors = rows[~rows.correct].sort_values("confidence", ascending=False)
    errors.to_csv(output / "validation_errors.csv", index=False)
    confusion = confusion_matrix(labels, prediction, labels=list(range(len(class_names))))
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(output / "confusion_matrix.csv")
    normalized = confusion / np.maximum(1, confusion.sum(1, keepdims=True))
    fig, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(class_names)), class_names, rotation=65, ha="right", fontsize=8)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=8)
    axis.set_xlabel("Prediksi")
    axis.set_ylabel("Label validasi")
    axis.set_title("Confusion matrix / normalisasi per kelas aktual")
    fig.colorbar(image, ax=axis, fraction=.046)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    pairs = [(int(confusion[a, b]), class_names[a], class_names[b])
             for a in range(len(class_names)) for b in range(len(class_names)) if a != b and confusion[a, b]]
    pd.DataFrame(sorted(pairs, reverse=True), columns=["count", "actual", "predicted"]).to_csv(output / "top_confusions.csv", index=False)
    if len(errors):
        chosen = errors.head(8)
        fig, axes = plt.subplots(2, 4, figsize=(12, 8))
        reference = frame.set_index("id")
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, chosen.itertuples(index=False)):
            source = reference.loc[row.id]
            image, _ = open_document(source.resolved_path, source.zip_member)
            image.thumbnail((320, 420))
            axis.imshow(image)
            axis.set_title(f"True: {class_names[row.label]}\nPred: {class_names[row.prediction]} ({row.confidence:.2f})", fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "high_confidence_errors.png", dpi=110)
        plt.close(fig)
    return metrics


def plot_history(run_dir):
    """Plot measured validation metrics and the recorded loss curves."""
    frame = pd.read_csv(Path(run_dir) / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(frame.epoch, frame.val_accuracy, label="Validation accuracy")
    axes[0].plot(frame.epoch, frame.val_macro_f1, label="Validation macro-F1")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[1].plot(frame.epoch, frame.train_loss, label="Train loss (augmentation/smoothing)")
    axes[1].plot(frame.epoch, frame.val_log_loss, label="Validation cross entropy")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "evaluation/training_curves.png", dpi=140)
    plt.close(fig)


def evaluate_model(project, run_dir):
    """Compare fixed single-scale vs multi-scale policies on the same holdout."""
    project, run_dir = Path(project), Path(run_dir)
    train, valid, test, sample, manifest = load_prepared(project)
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    evaluation_source = source_fingerprint()
    training_finished = json.loads((run_dir / "summary.json").read_text())["finished"]
    output = run_dir / "evaluation"
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        required = ["validation.npz", "test.npz", "submission.csv", "metrics.json"]
        if (cached["checkpoint_sha256"] == checkpoint_hash and cached["fingerprint"] == manifest["fingerprint"]
                and cached.get("evaluation_source") == evaluation_source
                and all((output / name).is_file() for name in required)):
            cached["training_finished"] = training_finished
            write_json(metadata_path, cached)
            plot_history(run_dir)
            print("Evaluasi cocok dengan checkpoint; memakai hasil tersimpan:", output, flush=True)
            return output
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint, device = load_inference(checkpoint_path)
    if checkpoint["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Checkpoint/data fingerprint berbeda")
    cfg = checkpoint["config"]
    torch.set_num_threads(cfg["cpu_threads"])
    sizes = list(dict.fromkeys([cfg["val_size"], *cfg["eval_sizes"]]))
    val_by_size = {}
    ids = list(valid.id)
    for size in sizes:
        print(f"Validasi {cfg['model']} pada {size}px", flush=True)
        probabilities, labels, predicted_ids = predict_loader(model, make_loader(valid, cfg, size), device, cfg["amp"])
        if predicted_ids != ids:
            raise ValueError("Urutan ID validasi berubah")
        val_by_size[size] = probabilities
    base = val_by_size[cfg["val_size"]]
    tta_sizes = list(dict.fromkeys(cfg["eval_sizes"]))
    tta = np.mean([val_by_size[size] for size in tta_sizes], axis=0)
    base_metrics, tta_metrics = metric_dict(labels, base), metric_dict(labels, tta)
    use_tta = (tta_metrics["accuracy"], -tta_metrics["log_loss"]) > (base_metrics["accuracy"], -base_metrics["log_loss"])
    selected_sizes = tta_sizes if use_tta else [cfg["val_size"]]
    selected = tta if use_tta else base
    write_json(output / "tta_comparison.json", {"single_scale": base_metrics, "multi_scale": tta_metrics,
               "candidate_tta_sizes": tta_sizes, "selected_sizes": selected_sizes,
               "caution": "Policy selected on validation: this is a selection score, not an unbiased final test."})
    metrics = report_predictions(valid, selected, manifest["class_names"], output)
    np.savez_compressed(output / "validation.npz", ids=np.asarray(ids, dtype=str), labels=labels, probabilities=selected)
    test_predictions = []
    for size in selected_sizes:
        print(f"Prediksi test {cfg['model']} pada {size}px", flush=True)
        probabilities, _, test_ids = predict_loader(model, make_loader(test, cfg, size), device, cfg["amp"])
        if test_ids != list(test.id):
            raise ValueError("Urutan ID test berubah")
        test_predictions.append(probabilities)
    test_probability = np.mean(test_predictions, axis=0)
    np.savez_compressed(output / "test.npz", ids=np.asarray(test.id, dtype=str), probabilities=test_probability)
    submission = build_submission(test, sample, test.id, test_probability)
    submission.to_csv(output / "submission.csv", index=False)
    # A clean, augmentation-free sample is more interpretable than mixup accuracy.
    clean = train.sample(n=min(len(train), cfg["clean_train_eval_rows"]), random_state=cfg["seed"])
    clean_probability, clean_labels, _ = predict_loader(model, make_loader(clean, cfg, cfg["val_size"]), device, cfg["amp"])
    clean_metrics = metric_dict(clean_labels, clean_probability)
    write_json(output / "generalization_diagnostics.json", {
        "clean_train_sample_rows": len(clean), "clean_train_single_scale": clean_metrics,
        "validation_single_scale": base_metrics,
        "accuracy_gap": clean_metrics["accuracy"] - base_metrics["accuracy"],
        "interpretation": "A large positive gap can indicate overfitting; consider learning curves, class errors, and sample uncertainty together.",
    })
    metadata = {
        "model": cfg["model"], "epoch": checkpoint["epoch"], "weight_type": checkpoint["weight_type"],
        "checkpoint_sha256": checkpoint_hash, "fingerprint": manifest["fingerprint"],
        "evaluation_source": evaluation_source,
        "selected_sizes": selected_sizes, "metrics": metrics,
        "submission_sha256": sha256_file(output / "submission.csv"),
        "training_finished": training_finished,
    }
    write_json(metadata_path, metadata)
    plot_history(run_dir)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def compare_models(project, evaluation_dirs):
    """Evaluate single models and a pre-specified 50:50 probability ensemble."""
    if not 1 <= len(evaluation_dirs) <= 2:
        raise ValueError("Supply one or two evaluated models")
    project = Path(project)
    train, valid, test, sample, manifest = load_prepared(project)
    directory = project / "outputs/comparison"
    directory.mkdir(parents=True, exist_ok=True)
    candidates, records = {}, []
    for evaluation_dir in map(Path, evaluation_dirs):
        meta = json.loads((evaluation_dir / "metadata.json").read_text())
        if meta["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Ensemble requires the exact same validation split and data")
        with np.load(evaluation_dir / "validation.npz", allow_pickle=False) as data:
            ids, labels, val_probability = data["ids"], data["labels"], data["probabilities"]
        with np.load(evaluation_dir / "test.npz", allow_pickle=False) as data:
            test_ids, test_probability = data["ids"], data["probabilities"]
        if list(ids) != list(valid.id) or list(test_ids) != list(test.id) or not np.array_equal(labels, valid.label.to_numpy()):
            raise ValueError("Prediction rows do not align; refusing unsafe blending")
        name = meta["model"]
        if name in candidates:
            raise ValueError("Duplicate architecture names in comparison")
        candidates[name] = (val_probability, test_probability)
    if len(candidates) == 2:
        candidates["ensemble_equal"] = tuple(np.mean([values[index] for values in candidates.values()], axis=0) for index in [0, 1])
    for name, (val_probability, test_probability) in candidates.items():
        metrics = metric_dict(valid.label.to_numpy(dtype=int), val_probability)
        records.append({"model": name, **metrics})
        submission = build_submission(test, sample, test.id, test_probability)
        submission.to_csv(directory / f"submission_{name}.csv", index=False)
    comparison = pd.DataFrame(records).sort_values(["accuracy", "log_loss"], ascending=[False, True])
    comparison.to_csv(directory / "model_comparison.csv", index=False)
    winner = comparison.iloc[0]["model"]
    val_probability, test_probability = candidates[winner]
    build_submission(test, sample, test.id, test_probability).to_csv(project / "submission.csv", index=False)
    report_predictions(valid, val_probability, manifest["class_names"], directory / "selected_diagnostics")
    provenance = {
        "selected_model": winner, "fingerprint": manifest["fingerprint"],
        "selection_rule": "highest validation accuracy; lower log-loss breaks ties",
        "ensemble_weights": [0.5, 0.5] if winner == "ensemble_equal" else None,
        "members": [json.loads((Path(path) / "metadata.json").read_text()) for path in evaluation_dirs],
        "score_type": "selection validation, not Kaggle public/private or an independent final test",
        "submission_sha256": sha256_file(project / "submission.csv"),
    }
    write_json(project / "submission_manifest.json", provenance)
    print(comparison.to_string(index=False), flush=True)
    print("Terpilih:", winner, "->", project / "submission.csv", flush=True)
    return comparison


def export_review(project):
    """Package diagnostics and manifests, excluding images and model checkpoints."""
    project = Path(project)
    target = project / "docvision_v2_review.zip"
    allowed = {".json", ".csv", ".png", ".npz"}
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for base in [project / "reports", project / "outputs"]:
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix in allowed:
                    archive.write(path, path.relative_to(project))
        for name in ["submission.csv", "submission_manifest.json", "data/splits/manifest.json"]:
            if (project / name).is_file():
                archive.write(project / name, name)
    return target
