"""Small CLI for a project exported by the self-contained notebook."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docvision_v2.data import find_data_root, prepare_data
from docvision_v2.engine import train_model
from docvision_v2.evaluate import compare_models, evaluate_model
from docvision_v2.reporting import export_github_results, update_results
from docvision_v2.runtime import probe_cuda, runtime_diagnostics


def main():
    """Run one workflow action after checking its required arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "compare", "report", "export", "diagnose"])
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-root")
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--evaluation-dir", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(find_data_root(args.data_root), args.project, extra_roots=args.image_root)
    elif args.command == "train":
        if args.config is None:
            parser.error("--config is required")
        train_model(args.project, json.loads(args.config.read_text()), args.run_name, args.max_hours)
    elif args.command == "evaluate":
        if args.run_dir is None:
            parser.error("--run-dir is required")
        evaluate_model(args.project, args.run_dir)
    elif args.command in {"compare", "report"}:
        if not args.evaluation_dir:
            parser.error("supply one --evaluation-dir per model")
        if args.command == "compare":
            compare_models(args.project, args.evaluation_dir)
        else:
            print(update_results(args.project, args.evaluation_dir))
    elif args.command == "export":
        print(export_github_results(args.project))
    else:
        print(json.dumps(runtime_diagnostics(), indent=2))
        print(json.dumps(probe_cuda(), indent=2))


if __name__ == "__main__":
    main()
