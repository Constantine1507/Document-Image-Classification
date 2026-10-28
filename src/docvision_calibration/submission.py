"""Generate a geometric ensemble from aligned PyTorch classifier probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import align, file_sha256, write_json
from .power import PowerMeanPool


def export_geometric_submission(convnext_path, efficientnet_path, sample_path, output):
    """Combine two NPZ files by ID and write a CSV without inventing a Kaggle score.

    Each NPZ must contain ``ids`` and ``probabilities`` with 16 class columns.
    The template supplies ID order only; its label values are never model inputs.
    """
    paths = [Path(convnext_path), Path(efficientnet_path), Path(sample_path)]
    output = Path(output)
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError('Output must differ from all input files')
    sample = pd.read_csv(sample_path, dtype={'id': str}, keep_default_na=False)
    if list(sample.columns) != ['id', 'label']:
        raise ValueError('Expected a submission template with columns id,label')
    if sample.empty or sample.id.duplicated().any() or sample.id.str.strip().eq('').any():
        raise ValueError('Template IDs must be nonempty and unique')
    ids = sample.id.to_numpy(dtype=str)
    probability = np.stack([align(path, ids) for path in paths[:2]], axis=1)
    with torch.inference_mode():
        model = PowerMeanPool(weight=0.5, power=0.0).eval()
        pooled = model(torch.as_tensor(probability, dtype=torch.float64)).numpy()
    result = pd.DataFrame({'id': ids, 'label': pooled.argmax(axis=1).astype(np.int64)})
    if not np.array_equal(result.id.to_numpy(), ids) or not result.label.between(0, 15).all():
        raise AssertionError('Submission schema or alignment changed')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    metadata = {
        'method': 'geometric_mean_50_50',
        'framework': 'PyTorch',
        'torch_version': str(torch.__version__),
        'rows': len(result),
        'columns': list(result.columns),
        'submission_sha256': file_sha256(output),
        'input_sha256': {
            'convnext_npz': file_sha256(convnext_path),
            'efficientnet_npz': file_sha256(efficientnet_path),
            'sample_submission': file_sha256(sample_path),
        },
        'class_order': list(range(16)),
        'kaggle_score': None,
        'note': 'A new CSV has no measured Kaggle score until evaluated by Kaggle.',
    }
    write_json(output.with_suffix('.manifest.json'), metadata)
    return metadata
