"""Append geometric pooling to the existing v2 training and evaluation workflow."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from docvision_calibration.submission import export_geometric_submission


def main():
    """Read explicitly selected evaluation files and generate aligned test labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--convnext', type=Path, required=True, help='ConvNeXt evaluation/test.npz')
    parser.add_argument('--efficientnet', type=Path, required=True, help='EfficientNet evaluation/test.npz')
    parser.add_argument('--sample-submission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = export_geometric_submission(
        args.convnext, args.efficientnet, args.sample_submission, args.output
    )
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
