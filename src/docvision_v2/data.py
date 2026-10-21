"""Read existing files/ZIP members in place, audit once, and freeze a split.

No image is copied, linked, uploaded, or extracted by this module. Metadata
and audit reports are written only beneath the chosen project directory.
"""
import hashlib
import io
import json
import os
import sqlite3
import threading
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .utils import digest_json, sha256_file, write_json

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder", "form", "handwritten",
    "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
    "resume", "scientific_publication", "scientific_report", "specification",
]
IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATH_COLUMNS = ("filename", "image_path", "path", "filepath", "file_path")
_ZIP_HANDLES = {}


def find_data_root(explicit=None):
    """Prefer the existing clean directory; otherwise discover image CSVs only."""
    if explicit:
        roots = [Path(explicit)]
    else:
        clean = Path("/kaggle/working/document_image_classification_clean")
        if all((clean / name).is_file() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            print("Menggunakan dataset bersih yang sudah tersedia:", clean, flush=True)
            return clean
        input_root = Path("/kaggle/input")
        roots = sorted({p.parent for p in input_root.rglob("train.csv")}) if input_root.exists() else []
    candidates = []
    for root in roots:
        if not all((root / name).is_file() for name in ("train.csv", "test.csv", "sample_submission.csv")):
            continue
        columns = pd.read_csv(root / "train.csv", nrows=0).columns
        if "label" in columns and any(column in columns for column in PATH_COLUMNS):
            candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(
            "Pilih DATA_ROOT yang berisi train.csv, test.csv, sample_submission.csv, "
            "dan metadata gambar. Dataset tabular tidak sesuai. Kandidat: "
            + repr([str(p) for p in candidates])
        )
    return candidates[0]


def read_tables(root, expected_classes=16):
    """Normalize filename/image_path schemas while preserving submission IDs."""
    root = Path(root)
    tables = []
    for subset in ("train", "test"):
        frame = pd.read_csv(root / f"{subset}.csv", dtype=str, keep_default_na=False)
        path_column = next((c for c in PATH_COLUMNS if c in frame), None)
        if path_column is None or frame.empty:
            raise ValueError(f"{subset}.csv harus memiliki kolom filename/image_path dan tidak kosong.")
        frame["image_path"] = frame[path_column].str.replace("\\", "/", regex=False)
        if "id" not in frame:
            frame["id"] = frame["image_path"]
        if frame.id.eq("").any() or frame.id.duplicated().any() or frame.image_path.eq("").any():
            raise ValueError(f"ID/path kosong atau ID duplikat dalam {subset}.csv")
        frame["subset"] = subset
        if subset == "train":
            labels = pd.to_numeric(frame["label"], errors="raise")
            if not labels.eq(labels.astype(int)).all():
                raise ValueError("Label harus integer.")
            frame["label"] = labels.astype(int)
            if set(frame.label) != set(range(expected_classes)):
                raise ValueError(f"Label harus lengkap 0..{expected_classes-1}; jangan mengganti mapping kelas.")
        elif "label" in frame:
            frame = frame.drop(columns="label")
        tables.append(frame)
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"id": str}, keep_default_na=False)
    if list(sample.columns) != ["id", "label"]:
        raise ValueError("Template kompetisi ini harus memiliki kolom id,label.")
    sample["id"] = sample.id.astype(str)
    if sample.id.duplicated().any() or sample.id.eq("").any() or set(sample.id) != set(tables[1].id):
        raise ValueError("ID sample_submission berbeda dengan test.csv.")
    return tables[0], tables[1], sample


class ImageResolver:
    """Resolve exact paths first; use a unique suffix or ZIP member as fallback."""
    def __init__(self, roots, allow_zip=True):
        """Register search roots and optional direct ZIP access."""
        self.roots = list(dict.fromkeys(Path(root).resolve() for root in roots))
        self.allow_zip = allow_zip
        self.index = None

    def _index(self):
        """Index basenames lazily across files and ZIP members."""
        self.index = defaultdict(list)
        seen = set()
        for root in self.roots:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    identity = (str(path.resolve()), "")
                    if identity not in seen:
                        seen.add(identity)
                        self.index[path.name].append((path.as_posix(), *identity))
                elif self.allow_zip and path.suffix.lower() == ".zip":
                    with ZipFile(path) as archive:
                        for name in archive.namelist():
                            if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES:
                                identity = (str(path.resolve()), name)
                                if identity not in seen:
                                    seen.add(identity)
                                    self.index[PurePosixPath(name).name].append((name, *identity))

    def resolve(self, filename):
        """Resolve one safe image path and reject ambiguous matches."""
        relative = PurePosixPath(str(filename).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"CSV harus memakai path relatif aman: {filename}")
        for root in self.roots:
            direct = root / str(relative)
            if direct.is_file():
                return str(direct.resolve()), ""
        if self.index is None:
            self._index()
        candidates = self.index.get(relative.name, [])
        for length in range(len(relative.parts), 0, -1):
            suffix = relative.parts[-length:]
            matches = {(path, member) for display, path, member in candidates
                       if PurePosixPath(display).parts[-length:] == suffix}
            existing_files = {item for item in matches if not item[1]}
            if existing_files:
                matches = existing_files
            if len(matches) == 1:
                return next(iter(matches))
            if len(matches) > 1:
                raise ValueError(f"Path ambigu: {filename}; cocok dengan {len(matches)} file.")
        raise FileNotFoundError(f"Gambar tidak ditemukan: {filename}; roots={self.roots}")


def zip_handle(path):
    """Reuse one read-only ZIP handle per process and thread."""
    key = (os.getpid(), threading.get_ident(), str(path))
    if key not in _ZIP_HANDLES:
        _ZIP_HANDLES[key] = ZipFile(path)
    return _ZIP_HANDLES[key]


def open_document(path, member=""):
    """Decode first page and EXIF orientation consistently in audit/train/test."""
    source = io.BytesIO(zip_handle(path).read(member)) if member else path
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Corrupt EXIF data.*", category=UserWarning)
        with Image.open(source) as image:
            frames = getattr(image, "n_frames", 1)
            image.seek(0)
            image = ImageOps.exif_transpose(image).convert("RGB").copy()
    return image, frames


def file_signature(path, member=""):
    """Fingerprint file metadata and ZIP CRC for audit cache lookup."""
    stat = Path(path).stat()
    record = [str(path), int(stat.st_size), int(stat.st_mtime_ns), member]
    if member:
        info = zip_handle(path).getinfo(member)
        record += [info.CRC, info.file_size]
    return digest_json(record)


def inspect_image(record):
    """Hash decoded pixels; errors become explicit audit rows, never fake images."""
    path, member, signature = record
    try:
        image, frames = open_document(path, member)
        digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
        digest.update(image.tobytes())
        return signature, {"pixel_sha256": digest.hexdigest(), "width": image.width,
                           "height": image.height, "frames": frames, "error": ""}
    except Exception as exc:
        return signature, {"pixel_sha256": "", "width": 0, "height": 0, "frames": 0,
                           "error": f"{type(exc).__name__}: {exc}"}


def audit_images(frame, cache_path, workers=4):
    """Reuse SQLite hashes for unchanged files; audit new files in parallel."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records, resolved = [], []
    for row in frame.itertuples(index=False):
        signature = file_signature(row.resolved_path, row.zip_member)
        records.append((row.resolved_path, row.zip_member, signature))
    with sqlite3.connect(cache_path) as database:
        database.execute("CREATE TABLE IF NOT EXISTS audit (signature TEXT PRIMARY KEY, payload TEXT)")
        cached = {key: json.loads(value) for key, value in database.execute("SELECT signature,payload FROM audit")}
        pending = list({r[2]: r for r in records if r[2] not in cached}.values())
        print(f"Audit: {len(records):,} gambar, {len(records)-len(pending):,} tersedia di cache, "
              f"{len(pending):,} perlu dibaca. Tidak ada salinan gambar.", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            for index, (key, value) in enumerate(executor.map(inspect_image, pending), 1):
                cached[key] = value
                database.execute("INSERT OR REPLACE INTO audit VALUES (?,?)", (key, json.dumps(value)))
                if index % 500 == 0 or index == len(pending):
                    database.commit()
                    print(f"Audit baru: {index:,}/{len(pending):,}", flush=True)
        for path, member, key in records:
            resolved.append(dict(cached[key], file_signature=key))
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(resolved)], axis=1)


def stratified_group_holdout(frame, fraction=0.2, seed=42):
    """Keep exact duplicate pixels together, balancing class counts approximately."""
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be inside (0,1)")
    rng = np.random.default_rng(seed)
    validation_groups = set()
    for label, rows in frame.groupby("label", sort=True):
        counts = rows.groupby("pixel_sha256", sort=True).size()
        groups = list(counts.index)
        if len(groups) < 2:
            raise ValueError(f"Kelas {label} membutuhkan minimal dua kelompok gambar independen.")
        rng.shuffle(groups)
        count, target = 0, max(1, round(len(rows) * fraction))
        for group in groups[:-1]:
            validation_groups.add(group)
            count += int(counts[group])
            if count >= target:
                break
    mask = frame.pixel_sha256.isin(validation_groups)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def verify_split(train, valid, full):
    """Reject missing IDs, overlapping pixel groups or absent classes."""
    if set(train.id) & set(valid.id) or set(train.id) | set(valid.id) != set(full.id):
        raise ValueError("ID split tumpang tindih atau tidak mencakup data bersih.")
    if set(train.pixel_sha256) & set(valid.pixel_sha256):
        raise ValueError("Kebocoran: piksel identik masuk training dan validasi.")
    for subset in [train, valid]:
        if set(subset.label) != set(full.label):
            raise ValueError("Ada kelas yang tidak terwakili pada salah satu split.")


def compatible_existing_split(directory, full):
    """Reuse a previous split only when IDs, paths, labels and hash groups match."""
    directory = Path(directory)
    if not all((directory / name).is_file() for name in ["train.csv", "val.csv"]):
        return None
    outputs = []
    reference = full.set_index("id")
    for name in ["train.csv", "val.csv"]:
        old = pd.read_csv(directory / name, dtype={"id": str})
        if old.id.duplicated().any() or not set(old.id).issubset(reference.index):
            return None
        new = reference.loc[old.id].reset_index()
        old_path = "image_path" if "image_path" in old else "filename"
        if old_path not in old or "label" not in old:
            return None
        if list(new.image_path) != list(old[old_path]) or list(new.label) != list(old.label.astype(int)):
            return None
        outputs.append(new)
    try:
        verify_split(*outputs, full)
    except ValueError:
        return None
    return outputs


def prepare_data(root, project, extra_roots=(), allow_zip=True, workers=4,
                 seed=42, val_fraction=0.2, legacy_split=None, expected_classes=16):
    """Create metadata-only manifests; quarantine corrupt/conflicting train rows."""
    root, project = Path(root).resolve(), Path(project).resolve()
    report_dir, split_dir = project / "reports", project / "data/splits"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample = read_tables(root, expected_classes)
    resolver = ImageResolver([root, *extra_roots], allow_zip)
    path_rows, path_errors = [], []
    for frame in [train, test]:
        for row in frame.to_dict("records"):
            try:
                path, member = resolver.resolve(row["image_path"])
                path_rows.append(dict(row, resolved_path=path, zip_member=member))
            except Exception as exc:
                path_errors.append(dict(row, reason=f"path_error: {exc}"))
    pd.DataFrame(path_errors).to_csv(report_dir / "missing_paths.csv", index=False)
    if path_errors:
        if any(row["subset"] == "test" for row in path_errors) or len(path_errors) > max(2, .01 * len(train)):
            raise ValueError("Path dataset belum benar. Lihat reports/missing_paths.csv; tidak ada file sumber diubah.")
    audit = audit_images(pd.DataFrame(path_rows), project / "data/audit_cache.sqlite3", workers)
    audit.to_csv(report_dir / "image_audit.csv", index=False)
    bad = audit[audit.error.ne("")].copy()
    bad.to_csv(report_dir / "unreadable_images.csv", index=False)
    if bad.subset.eq("test").any() or len(bad) > max(2, .01 * len(train)):
        raise ValueError("Gambar test rusak atau terlalu banyak gambar gagal. Periksa unreadable_images.csv.")
    readable = audit[audit.error.eq("")].copy()
    full = readable[readable.subset.eq("train")].copy()
    full["label"] = full.label.astype(int)
    test = readable[readable.subset.eq("test")].copy().drop(columns="label", errors="ignore")
    conflicts = full.groupby("pixel_sha256").label.nunique()
    conflict_hashes = set(conflicts[conflicts.gt(1)].index)
    quarantine = full[full.pixel_sha256.isin(conflict_hashes)].copy()
    quarantine.to_csv(report_dir / "conflicting_labels_quarantined.csv", index=False)
    full = full[~full.pixel_sha256.isin(conflict_hashes)].copy()
    if set(full.label) != set(range(expected_classes)):
        raise ValueError("Pembersihan menghilangkan satu kelas. Perlu pemeriksaan manual.")
    overlap = set(full.pixel_sha256) & set(test.pixel_sha256)
    readable[readable.pixel_sha256.isin(overlap)].to_csv(report_dir / "train_test_exact_overlap.csv", index=False)
    existing = compatible_existing_split(split_dir, full)
    split_source = "v2 existing verified split"
    if existing is None and legacy_split:
        existing = compatible_existing_split(legacy_split, full)
        split_source = "legacy verified split" if existing else "new exact-group stratified holdout"
    if existing is None:
        existing = stratified_group_holdout(full, val_fraction, seed)
        split_source = "new exact-group stratified holdout"
    train_part, valid_part = existing
    verify_split(train_part, valid_part, full)
    train_part.to_csv(split_dir / "train.csv", index=False)
    valid_part.to_csv(split_dir / "val.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    sample.to_csv(split_dir / "sample_submission.csv", index=False)
    full.to_csv(split_dir / "full_train.csv", index=False)
    fingerprint = digest_json({
        "rows": full.sort_values("id")[["id", "image_path", "label", "pixel_sha256"]].to_dict("records"),
        "train_ids": sorted(train_part.id), "valid_ids": sorted(valid_part.id),
        "test": test.sort_values("id")[["id", "image_path", "pixel_sha256"]].to_dict("records"),
        "sample_ids": list(sample.id),
    })
    class_names = CLASS_NAMES[:expected_classes]
    if "class_name" in full and full.groupby("label").class_name.nunique().eq(1).all():
        class_names = list(full.groupby("label", sort=True).class_name.first())
    manifest = {
        "fingerprint": fingerprint, "data_root": str(root), "class_names": class_names,
        "train_rows": len(train_part), "valid_rows": len(valid_part), "test_rows": len(test),
        "split_source": split_source, "seed": seed, "requested_validation_fraction": val_fraction,
        "actual_validation_fraction": len(valid_part) / len(full),
        "conflicting_rows_excluded": len(quarantine), "corrupt_rows_excluded": len(bad),
        "missing_train_rows_excluded": len(path_errors),
        "duplicate_train_rows": int(full.pixel_sha256.duplicated().sum()),
        "train_test_overlap_groups": len(overlap), "multipage_images": int(readable.frames.gt(1).sum()),
        "storage_mode": "existing files / ZIP members, no image copies or extraction",
        "limitation": "Exact pixel groups only. Near duplicates and related source documents may remain.",
    }
    write_json(split_dir / "manifest.json", manifest)
    partition_rows = pd.concat([train_part.assign(partition="train"), valid_part.assign(partition="valid")], ignore_index=True)
    pd.crosstab(partition_rows.label, partition_rows.partition).to_csv(report_dir / "split_class_counts.csv")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def load_prepared(project):
    """Read prepared split metadata and preserve string identifiers."""
    root = Path(project) / "data/splits"
    manifest = json.loads((root / "manifest.json").read_text())
    tables = [pd.read_csv(root / name, dtype={"id": str}, keep_default_na=False)
              for name in ("train.csv", "val.csv", "test.csv", "sample_submission.csv")]
    return *tables, manifest


def build_submission(test, sample, ids, probabilities):
    """Join by IDs; never sort test predictions blindly or copy known labels."""
    ids = list(map(str, ids))
    probabilities = np.asarray(probabilities)
    if len(ids) != len(test) or len(set(ids)) != len(ids) or set(ids) != set(test.id):
        raise ValueError("Prediction IDs mismatch test metadata")
    if probabilities.ndim != 2 or len(probabilities) != len(ids) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probabilities")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-4):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    prediction = pd.DataFrame({"id": ids, "label": probabilities.argmax(1).astype(int)})
    result = sample[["id"]].merge(prediction, on="id", how="left", validate="one_to_one", sort=False)
    if result.isna().any().any() or list(result.id) != list(sample.id):
        raise ValueError("Submission alignment failed")
    return result[["id", "label"]]
