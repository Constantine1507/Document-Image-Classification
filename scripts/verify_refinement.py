"""Independently verify the final CSVs against the original NPZ probabilities.

This script deliberately does not import the PyTorch pooling implementation.
NumPy formulas check label alignment and numerical equivalence independently.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def read_predictions(archive, member, ids):
    """Read one matrix from the original ZIP and align by unique sample IDs."""
    with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as data:
        original_ids = data['ids'].astype(str)
        assert len(np.unique(original_ids)) == len(original_ids)
        assert set(original_ids) == set(ids)
        positions = pd.Index(original_ids).get_indexer(ids)
        probabilities = data['probabilities'][positions].astype(np.float64)
        labels = data['labels'][positions] if 'labels' in data else None
    assert probabilities.shape == (len(ids), 16)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    return probabilities, labels


def pooled(probabilities, kind):
    """Compute the mathematical pooling rule directly using NumPy."""
    first, second = (np.clip(p, 1e-12, 1) for p in probabilities)
    if kind == 'geometric':
        result = np.sqrt(first * second)
    else:
        result = 1 / (0.35 / first + 0.65 / second)
    return result / result.sum(axis=1, keepdims=True)


def main():
    """Check schema, order, all predictions, metrics, hashes and source syntax."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-zip', type=Path, required=True)
    parser.add_argument('--final-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.final_dir / 'submission_manifest.json').read_text())
    checksum = hashlib.sha256(args.review_zip.read_bytes()).hexdigest()
    assert checksum == manifest['data_provenance']['archive_sha256']
    checked = []
    per_class = []
    with ZipFile(args.review_zip) as archive:
        sample = pd.read_csv(io.BytesIO(archive.read('submission.csv')), dtype={'id': str})
        paths = [
            'outputs/convnext_tiny_v2_gpu_fix_seed42/evaluation/',
            'outputs/efficientnet_v2_s_v2_gpu_fix_seed42/evaluation/',
        ]
        assert len(sample) == 7994 and sample.id.is_unique
        test = [read_predictions(archive, p + 'test.npz', sample.id)[0] for p in paths]
        baseline = np.mean(test, axis=0).argmax(axis=1)
        assert np.array_equal(baseline, sample.label.to_numpy())
        valid_ids = np.load(args.final_dir / 'submission_validation.npz', allow_pickle=False)['ids']
        valid_data = [read_predictions(archive, p + 'validation.npz', valid_ids) for p in paths]
        valid = [item[0] for item in valid_data]
        labels = valid_data[0][1]
        assert np.array_equal(labels, valid_data[1][1])
        baseline_valid = np.mean(valid, axis=0).argmax(axis=1)
        assert int((baseline_valid == labels).sum()) == 5652
        names = manifest['data_provenance']['class_names']
        for role, kind in [('primary', 'geometric'), ('exploratory', 'harmonic')]:
            record = manifest[role]
            path = args.final_dir / record['file']
            frame = pd.read_csv(path, dtype={'id': str})
            assert list(frame.columns) == ['id', 'label']
            assert len(frame) == 7994 and frame.id.is_unique
            assert np.array_equal(frame.id.to_numpy(), sample.id.to_numpy())
            assert pd.api.types.is_integer_dtype(frame.label)
            assert frame.label.between(0, 15).all()
            probability = pooled(test, kind)
            assert np.array_equal(probability.argmax(axis=1), frame.label.to_numpy())
            changes = int((frame.label.to_numpy() != baseline).sum())
            assert changes == record['changed_test_labels']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == record['sha256']
            assert digest != manifest['data_provenance']['baseline_submission_sha256']
            with np.load(args.final_dir / f'{path.stem}_test.npz', allow_pickle=False) as stored:
                assert np.array_equal(stored['ids'], sample.id.to_numpy())
                difference = float(np.abs(probability - stored['probabilities']).max())
                assert difference < 1e-10
            valid_probability = pooled(valid, kind)
            prediction = valid_probability.argmax(axis=1)
            correct = int((prediction == labels).sum())
            assert correct == record['validation_metrics']['correct']
            for method, pred in [('baseline', baseline_valid), (role, prediction)]:
                if method == 'baseline' and role != 'primary':
                    continue
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, pred, labels=np.arange(16), zero_division=0
                )
                for index, name in enumerate(names):
                    per_class.append({'method': method, 'label': index, 'class_name': name,
                                      'precision': precision[index], 'recall': recall[index],
                                      'f1': f1[index], 'support': int(support[index])})
            checked.append({'file': path.name, 'rows': len(frame), 'sha256': digest,
                            'changed_test_labels': changes, 'independent_labels_match': True,
                            'maximum_probability_difference': difference,
                            'validation_correct': correct, 'validation_accuracy': correct / len(labels)})
    project = Path(__file__).resolve().parents[1]
    syntax_files = sorted(project.joinpath('src').rglob('*.py')) + sorted(project.joinpath('scripts').glob('*.py'))
    for path in syntax_files:
        compile(path.read_text(), str(path), 'exec')
    result = {'passed': True, 'archive_sha256': checksum, 'files': checked,
              'compiled_source_files': len(syntax_files), 'new_kaggle_score': None,
              'test_accuracy_measured': False,
              'verification_method': 'Independent NumPy formulas on original archived probabilities; no test targets.'}
    (args.report_dir / 'REPRODUCTION_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    pd.DataFrame(per_class).to_csv(args.report_dir / 'per_class_final_comparison.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
