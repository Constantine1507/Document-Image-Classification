"""Small serialization, reproducibility, and fingerprint helpers."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def sha256_file(path):
    """Hash a file without loading its complete contents into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    """Hash a canonical JSON representation for identity checks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    """Write JSON atomically; NaN is not permitted in reports."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))
    os.replace(temporary, path)


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch and available CUDA generators."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng():
    """Use tensors and primitive objects compatible with weights_only=True."""
    import torch
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore RNG states captured at an epoch boundary."""
    import torch
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_torch_save(path, payload):
    """Replace a checkpoint only after its temporary write completes."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def source_fingerprint():
    """Hash each package source file for strict experiment provenance."""
    root = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}
