"""Verify committed submission identity and schema without downloading the dataset."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    """Check both historical CSVs against their hashes, ID order and change count."""
    directory = ROOT / 'submissions'
    manifest = json.loads((directory / 'manifest.json').read_text())
    collected = {}
    checks = []
    reference_ids = None
    for record in manifest['submissions']:
        path = directory / record['file']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record['sha256']:
            raise AssertionError(f'CSV hash changed: {path.name}')
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['id', 'label']:
                raise AssertionError(f'Unexpected columns: {path.name}')
            rows = list(reader)
        ids = [row['id'] for row in rows]
        labels = [int(row['label']) for row in rows]
        if len(rows) != 7994 or len(ids) != len(set(ids)) or any(not value for value in ids):
            raise AssertionError(f'Unexpected or duplicated IDs: {path.name}')
        if any(label < 0 or label > 15 for label in labels):
            raise AssertionError(f'Invalid class label: {path.name}')
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise AssertionError('Submission ID order differs')
        collected[record['version']] = labels
        checks.append({'file': path.name, 'rows': len(rows), 'sha256': digest})
    changes = sum(a != b for a, b in zip(collected['baseline'], collected['latest']))
    if changes != manifest['changed_test_labels_latest_vs_baseline']:
        raise AssertionError('Unexpected changed prediction count')
    print(json.dumps({'status': 'passed', 'submissions': checks,
                      'changed_test_labels': changes}, indent=2))


if __name__ == '__main__':
    main()
