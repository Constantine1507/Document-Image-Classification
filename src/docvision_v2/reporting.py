"""Export aggregate experiment evidence and update the generated README section."""
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .utils import write_json


def update_results(project, evaluation_dirs):
    """Copy measured run summaries without dataset images or model weights."""
    project = Path(project)
    output = project / "results/latest"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(project / "outputs/comparison/model_comparison.csv")
    table.to_csv(output / "model_comparison.csv", index=False)
    manifest = json.loads((project / "submission_manifest.json").read_text())
    write_json(output / "submission_manifest.json", manifest)
    for directory in map(Path, evaluation_dirs):
        meta = json.loads((directory / "metadata.json").read_text())
        target = output / meta["model"]
        target.mkdir(exist_ok=True)
        for name in ["metrics.json", "metadata.json", "tta_comparison.json",
                     "generalization_diagnostics.json", "classification_report.csv",
                     "confusion_matrix.csv", "top_confusions.csv", "training_curves.png",
                     "confusion_matrix.png"]:
            if (directory / name).is_file():
                shutil.copy2(directory / name, target / name)
        for name in ["history.csv", "config.json", "summary.json", "environment.json"]:
            if (directory.parent / name).is_file():
                shutil.copy2(directory.parent / name, target / name)
    for source, name in [(project / "data/splits/manifest.json", "data_audit.json"),
                         (project / "reports/split_class_counts.csv", "split_class_counts.csv")]:
        if source.is_file():
            shutil.copy2(source, output / name)
    lines = ["## Hasil run terbaru", "", "Hasil berikut dihasilkan dari file evaluasi run ini.", "",
             "| Model | Validation accuracy | Macro-F1 | Log-loss |", "|---|---:|---:|---:|"]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.model} | {row.accuracy:.6f} | {row.macro_f1:.6f} | {row.log_loss:.6f} |")
    lines += ["", f"Pilihan validasi: **{manifest['selected_model']}**.",
              "Skor ini digunakan untuk seleksi model; skor Kaggle tidak diisi otomatis."]
    for member in manifest["members"]:
        diagnostic = output / member["model"] / "generalization_diagnostics.json"
        if diagnostic.is_file():
            value = json.loads(diagnostic.read_text())
            lines += ["", f"{member['model']}: selisih akurasi clean-train sample dan validasi "
                      f"{value['accuracy_gap'] * 100:.2f} poin persentase. "
                      "Interpretasikan bersama kurva loss dan kesalahan per kelas."]
    if not all(member["training_finished"] for member in manifest["members"]):
        lines += ["", "**Run masih dijeda/belum selesai; hasil ini bersifat sementara.**"]
    section = "\n".join(lines) + "\n"
    (output / "ANALYSIS.md").write_text(section)
    readme = project / "README.md"
    text = readme.read_text()
    start, end = "<!-- LATEST_RUN_START -->", "<!-- LATEST_RUN_END -->"
    if start in text and end in text:
        text = text.split(start)[0] + start + "\n" + section + end + text.split(end, 1)[1]
    else:
        text += "\n" + start + "\n" + section + end + "\n"
    readme.write_text(text)
    return output


def export_github_results(project):
    """Bundle the README and aggregate latest results, excluding row-level data."""
    project = Path(project)
    target = project / "github_results.zip"
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.write(project / "README.md", "README.md")
        for path in sorted((project / "results/latest").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(project))
    return target
