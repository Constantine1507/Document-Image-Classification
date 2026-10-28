"""Validate and align probabilities from a saved review ZIP; do not use image IDs as features."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def file_sha256(path):
    """Compute a stable SHA-256 for provenance and cache validation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Write finite, human-readable JSON atomically."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')
    temporary.replace(path)


def align(path, ids, labels=None):
    """Reorder a probability matrix by unique sample IDs and verify its labels."""
    ids=np.asarray(ids,dtype=str)
    if len(set(ids))!=len(ids):raise ValueError('Duplicate target IDs')
    with np.load(path,allow_pickle=False) as content:
        original=content['ids'].astype(str)
        if len(set(original))!=len(original) or set(ids)!=set(original):raise ValueError('Prediction IDs differ')
        order=pd.Index(original).get_indexer(ids)
        probability=content['probabilities'][order].astype(np.float64)
        if labels is not None and not np.array_equal(content['labels'][order],labels):raise ValueError('True labels differ')
    if probability.shape!=(len(ids),16) or not np.isfinite(probability).all():raise ValueError('Invalid probability dimensions')
    if (probability<0).any() or not np.allclose(probability.sum(1),1,atol=1e-5):raise ValueError('Invalid probability mass')
    return probability


def load_review(archive_path, extraction):
    """Extract the provided archive and verify that it reproduces the original baseline."""
    extraction=Path(extraction);extraction.mkdir(parents=True,exist_ok=True)
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path=(extraction/info.filename).resolve()
            if not path.is_relative_to(extraction.resolve()):raise ValueError('Unsafe archive path')
            if not info.is_dir():
                path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(info) as stream:path.write_bytes(stream.read())
    manifest=json.loads((extraction/'data/splits/manifest.json').read_text())
    sample=pd.read_csv(extraction/'submission.csv',dtype={'id':str},keep_default_na=False)
    if list(sample.columns)!=['id','label'] or len(sample)!=7994 or sample.id.duplicated().any():raise ValueError('Unexpected test template')
    folders={}
    for path in extraction.glob('outputs/*/evaluation/metadata.json'):
        meta=json.loads(path.read_text())
        if meta['model'] in {'convnext_tiny','efficientnet_v2_s'}:
            if meta['model'] in folders:raise ValueError('Ambiguous backbone evaluation')
            if meta['fingerprint']!=manifest['fingerprint'] or meta.get('full_data'):raise ValueError('Different validation scope')
            folders[meta['model']]=path.parent
    if len(folders)!=2:raise ValueError('Both backbone probability sets are required')
    names=['convnext_tiny','efficientnet_v2_s']
    valid=pd.read_csv(folders[names[0]]/'validation_predictions.csv',dtype={'id':str,'pixel_sha256':str},keep_default_na=False)
    if len(valid)!=6392 or valid.id.duplicated().any():raise ValueError('Unexpected validation IDs')
    if set(valid.label)!=set(range(16)):raise ValueError('Incorrect class labels')
    labels=valid.label.to_numpy(dtype=np.int64)
    probability=np.stack([align(folders[name]/'validation.npz',valid.id,labels) for name in names],axis=1)
    testing=np.stack([align(folders[name]/'test.npz',sample.id) for name in names],axis=1)
    if not np.array_equal(testing.mean(1).argmax(1),sample.label):raise ValueError('Archived submission does not match equal ensemble')
    recorded=json.loads((extraction/'outputs/comparison/selected_diagnostics/metrics.json').read_text())
    if np.sum(probability.mean(1).argmax(1)==labels)!=recorded['correct']:raise ValueError('Baseline validation score differs')
    metadata={'archive_sha256':file_sha256(archive_path),'fingerprint':manifest['fingerprint'],
              'class_names':manifest['class_names'],'member_names':names,'validation_rows':len(valid),
              'validation_unique_groups':valid.pixel_sha256.nunique(),'test_rows':len(sample),
              'baseline_correct':recorded['correct'],'baseline_accuracy':recorded['accuracy'],
              'baseline_submission_sha256':file_sha256(extraction/'submission.csv'),
              'backbone_metadata':[json.loads((folders[name]/'metadata.json').read_text()) for name in names]}
    return valid,sample,probability,testing,metadata
