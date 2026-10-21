"""Check repository syntax, docstrings, generated notebook and evidence identity."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Validate maintained source and report only checks actually performed."""
    files = sorted(p for base in ["src", "scripts", "tests"] for p in (ROOT / base).rglob("*.py"))
    definitions, missing = 0, []
    for path in files:
        source = path.read_text()
        compile(source, str(path), "exec")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions += 1
                if not ast.get_docstring(node):
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    if missing:
        raise ValueError(f"Missing docstrings: {missing}")
    notebook = nbformat.read(ROOT / "notebooks/Document_Classification_v2_2_Kaggle.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_count, embedded = 0, 0
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        code_count += 1
        if cell.source.startswith("%%writefile "):
            header, content = cell.source.split("\n", 1)
            path = ROOT / header.split(maxsplit=1)[1]
            assert path.read_text().strip() == content.strip(), str(path)
            embedded += 1
        else:
            compile(cell.source, f"notebook-cell-{code_count}", "exec")
        assert not cell.outputs and cell.execution_count is None
    subprocess.run([sys.executable, str(ROOT / "scripts/build_notebook.py"), "--check"], check=True)
    archive = ROOT / "notebooks/executed/v2_gpu_fix_kaggle.ipynb"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == "1af2fc26eabea894bfec9dbb18884ee8678b510ace3ee783927cfd34fdccb758"
    original = json.loads(archive.read_text())
    assert not any(o.get("output_type") == "error" for c in original["cells"] for o in c.get("outputs", []))
    ignored = set((ROOT / ".gitignore").read_text().splitlines())
    assert {"data/", "checkpoint/", "outputs/", "*.pt", "*.pth"} <= ignored
    maintained = [p for p in ROOT.rglob("*") if p.is_file() and not any(part in {".git", "__pycache__"} for part in p.parts)]
    largest = max(p.stat().st_size for p in maintained)
    assert largest < 100_000_000
    assert not any(p.suffix in {".pt", ".pth", ".ckpt"} for p in maintained)
    sys.path.insert(0, str(ROOT / "src"))
    from docvision_v2.models import default_config
    for name in ["convnext_tiny", "efficientnet_v2_s"]:
        assert json.loads((ROOT / "configs" / f"{name}.json").read_text()) == default_config(name)
    report = {"status": "passed", "python_files": len(files), "documented_definitions": definitions,
              "missing_docstrings": missing, "notebook_code_cells": code_count,
              "embedded_source_files": embedded, "archive_sha256": digest, "largest_file_bytes": largest}
    if len(sys.argv) > 1:
        output = Path(sys.argv[1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
