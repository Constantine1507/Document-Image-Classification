"""Build the self-contained Kaggle notebook from the reviewed source tree."""
import json
import hashlib
import sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks'
DEST.mkdir(exist_ok=True)
cells = []


def markdown(text):
    """Append a markdown cell to the generated notebook."""
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text, tag=None):
    """Append a code cell with an optional workflow tag."""
    cell = nbf.v4.new_code_cell(text.strip())
    if tag:
        cell.metadata['tags'] = [tag]
    cells.append(cell)


markdown('''
# Document Classification v2.2
## ConvNeXt Tiny + EfficientNetV2-S — baca dataset langsung

Notebook ini menggantikan alur ResNet18 yang berulang dengan satu pipeline:
**data yang sudah ada → audit tersimpan → split tetap → dua model → evaluasi → submission**.

- Tidak perlu mengunggah, menyalin, atau membuat symlink untuk setiap gambar.
- Jika file gambar sudah ada, file itu dipakai langsung. ZIP juga dapat dibaca tanpa ekstraksi.
- Semua source proyek disertakan; tidak perlu script tambahan dari Add Input.
- PyTorch dan torchvision; bobot awal ImageNet-1K resmi, bukan model RVL-CDIP yang berpotensi bocor.
- Run historis terarsip: ensemble validation accuracy 0,884230; skor Kaggle yang dilaporkan 0,88456. Notebook baru ini belum berisi hasil training baru.
- Pembaruan 28 Oktober: tahap 6b membuat kandidat geometric mean 50:50. Skor 0,88582 adalah laporan untuk CSV historis terbaru, bukan skor otomatis run baru.

**Langkah singkat:** aktifkan GPU → pastikan dataset gambar tersedia → sesuaikan sel pengaturan bila perlu → Run All.
Internet diperlukan untuk download pertama bobot pretrained, kecuali file bobot sudah disediakan lokal.
Untuk kualitas, jalankan kedua model; untuk uji awal atau waktu terbatas, isi `MODELS` dengan satu model dahulu.

Laporan analisis lengkap terdapat di bagian akhir dan pada `README.md` yang dibuat notebook.
''')
markdown('''
### 1. Pengaturan

`DATA_ROOT=None` memprioritaskan folder bersih sesi lama jika masih ada. Jika tidak,
notebook mencari dataset **gambar** di `/kaggle/input`; dataset structural-damage tabular tidak dipilih.
Jika ada beberapa dataset gambar, isi path secara eksplisit. `IMAGE_ROOTS` hanya diperlukan
jika metadata dan gambar berada di lokasi terpisah.

Penyimpanan `/kaggle/working` dari sesi yang telah berakhir tidak otomatis tersedia.
Pasang kembali dataset atau output notebook tersimpan melalui Add Input; tidak perlu menambahkan gambar satu per satu.
''')
code('''
from pathlib import Path

PROJECT = (Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()) / 'docvision_v2'
DATA_ROOT = None
IMAGE_ROOTS = []
ALLOW_ZIP_STREAMING = True
REUSE_LEGACY_SPLIT = True
AUDIT_WORKERS = 4
EXPECTED_CLASSES = 16
SPLIT_SEED = 42
VAL_FRACTION = 0.20

MODELS = ['convnext_tiny', 'efficientnet_v2_s']
RUN_TAG = 'v2_2'
REQUIRE_GPU = True
ENABLE_TRAINING = True
MAX_HOURS_PER_MODEL = None  # misalnya 4.5: jeda setelah epoch; ulang sel training untuk resume

# Contoh untuk GPU kecil: {'batch_size': 4, 'accumulation_steps': 8}
# Jangan mengubah config run yang sudah dimulai. Ganti RUN_TAG untuk eksperimen baru.
COMMON_OVERRIDES = {}
MODEL_OVERRIDES = {
    'convnext_tiny': {},
    'efficientnet_v2_s': {},
}
# Untuk offline, isi masing-masing override:
# {'local_imagenet_weights': '/kaggle/input/NAMA_BOBOT/convnext_tiny-983f1562.pth'}

# Opsional: salinan cache audit v2 dari output tersimpan (file metadata, bukan gambar).
AUDIT_CACHE_SOURCE = None
RUN_SMOKE_TESTS = False
''', 'user-config')
code('''
import os
import sys
import importlib
import importlib.util
import shutil

PROJECT = Path(PROJECT).resolve()
if str(PROJECT).startswith('/kaggle/input/'):
    raise ValueError('PROJECT harus berada di /kaggle/working, bukan input read-only.')
PROJECT.mkdir(parents=True, exist_ok=True)
for folder in ['src/docvision_v2', 'src/docvision_calibration', 'scripts', 'tests', 'configs', 'reports', 'data/splits', 'outputs', 'docs', 'results']:
    (PROJECT / folder).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
required = {'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
            'pandas': 'pandas', 'PIL': 'Pillow', 'sklearn': 'scikit-learn', 'matplotlib': 'matplotlib'}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    raise RuntimeError(f'Paket belum tersedia: {missing}. Gunakan runtime Kaggle standar; jangan mengganti pasangan torch/torchvision sembarangan.')

import torch
import torchvision
print('Project:', PROJECT)
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'tidak aktif')
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError('Aktifkan GPU pada Kaggle Settings / Accelerator, lalu jalankan ulang.')
''')
markdown('''
### 2. Source proyek yang lengkap

Sel berikut hanya menulis source, dokumentasi, dan pengujian; tidak menyalin gambar.
Sel `%%writefile` akan mengganti file source proyek yang namanya sama. Jika source sudah diedit
manual, simpan perubahan sebelum menjalankan kembali bagian ini. Resume memverifikasi hash source.
''')
for name in ['src/docvision_v2/__init__.py', 'src/docvision_v2/utils.py', 'src/docvision_v2/data.py', 'src/docvision_v2/models.py', 'src/docvision_v2/engine.py', 'src/docvision_v2/evaluate.py', 'src/docvision_v2/runtime.py', 'src/docvision_v2/reporting.py', 'scripts/run.py', 'tests/smoke_test.py', 'tests/runtime_test.py', 'requirements.txt', 'README.md', 'docs/METHODOLOGY.md', 'docs/TROUBLESHOOTING.md', 'docs/ASSIGNMENT_COMPLIANCE.md', 'configs/convnext_tiny.json', 'configs/efficientnet_v2_s.json']:
    code(f'%%writefile {name}\n' + (ROOT / name).read_text(), 'project-source')
for name in ['src/docvision_calibration/__init__.py', 'src/docvision_calibration/data.py', 'src/docvision_calibration/power.py', 'src/docvision_calibration/submission.py', 'scripts/ensemble_geometric.py', 'docs/PIPELINE_FLOW.md']:
    code(f'%%writefile {name}\n' + (ROOT / name).read_text(), 'project-source')
code('''
import json
import hashlib
import sys
import pandas as pd
from IPython.display import display, FileLink, Image as DisplayImage

sys.path.insert(0, str(PROJECT / 'src'))
importlib.invalidate_caches()
for module_name in list(sys.modules):
    if module_name in {'docvision_v2', 'docvision_calibration'} or module_name.startswith(('docvision_v2.', 'docvision_calibration.')):
        del sys.modules[module_name]

from docvision_v2.data import find_data_root, prepare_data, load_prepared
from docvision_v2.models import default_config
from docvision_v2.engine import train_model
from docvision_v2.evaluate import evaluate_model, compare_models, export_review
from docvision_v2.reporting import update_results, export_github_results
from docvision_v2.runtime import runtime_diagnostics, probe_cuda

CONFIGS = {}
for name in MODELS:
    config = default_config(name)
    config.update(num_classes=EXPECTED_CLASSES, require_gpu=REQUIRE_GPU)
    config.update(COMMON_OVERRIDES)
    config.update(MODEL_OVERRIDES.get(name, {}))
    CONFIGS[name] = config
    (PROJECT / 'configs' / f'{name}.json').write_text(json.dumps(config, indent=2))

display(pd.DataFrame([{key: cfg[key] for key in [
    'model', 'epochs', 'image_size', 'finetune_size', 'batch_size',
    'accumulation_steps', 'lr', 'seed']} for cfg in CONFIGS.values()]))
''')
markdown('''
### 3. Data langsung, audit cache, dan split

Audit pertama tetap perlu membaca piksel untuk mengenali gambar rusak dan duplikat.
Hasilnya disimpan, sehingga gambar yang tidak berubah tidak didekode lagi saat audit berikutnya.
Training tetap membaca minibatch gambar sesuai kebutuhan; ini bukan proses import/salin dataset.

Gambar training rusak dan kelompok duplikat dengan label konflik dikeluarkan **melalui metadata**.
Gambar sumber tidak dihapus. Gambar test yang rusak menghentikan proses agar submission tetap lengkap.
Split lama hanya dipakai jika cocok dengan data bersih dan bebas exact-duplicate overlap.
''')
code('''
DATA_ROOT = find_data_root(DATA_ROOT)
cache_target = PROJECT / 'data/audit_cache.sqlite3'
if AUDIT_CACHE_SOURCE and not cache_target.exists():
    shutil.copy2(Path(AUDIT_CACHE_SOURCE), cache_target)

legacy = Path('/kaggle/working/rvl_cdip_pytorch/data/splits') if REUSE_LEGACY_SPLIT else None
manifest = prepare_data(
    DATA_ROOT, PROJECT, extra_roots=IMAGE_ROOTS,
    allow_zip=ALLOW_ZIP_STREAMING, workers=AUDIT_WORKERS,
    seed=SPLIT_SEED, val_fraction=VAL_FRACTION,
    legacy_split=legacy, expected_classes=EXPECTED_CLASSES,
)
train_frame, valid_frame, test_frame, sample_submission, manifest = load_prepared(PROJECT)
display(pd.read_csv(PROJECT / 'reports/split_class_counts.csv'))
print('Asal split:', manifest['split_source'])
print('Train/valid/test:', len(train_frame), len(valid_frame), len(test_frame))
print('Exact train/test overlap:', manifest['train_test_overlap_groups'], '(dilaporkan; tidak dipakai untuk lookup label)')
''')
code('''
# Preview: hanya beberapa gambar, untuk memastikan path, orientasi, dan kelas masuk akal.
import matplotlib.pyplot as plt
from docvision_v2.data import open_document

preview = train_frame.groupby('label', sort=True).head(1).head(8)
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
for axis in axes.flat:
    axis.axis('off')
for axis, row in zip(axes.flat, preview.itertuples(index=False)):
    image, _ = open_document(row.resolved_path, row.zip_member)
    image.thumbnail((320, 420))
    axis.imshow(image)
    axis.set_title(manifest['class_names'][int(row.label)], fontsize=9)
fig.tight_layout()
fig.savefig(PROJECT / 'reports/class_preview.png', dpi=110)
plt.close(fig)
display(DisplayImage(filename=str(PROJECT / 'reports/class_preview.png')))
''')
markdown('''
### 4. Training dan resume

Default: maksimum 20 epoch/model; 320 piksel pada epoch 1–14 dan 384 pada epoch 15–20.
Validasi setiap epoch selalu pada 384 piksel agar checkpoint dapat dibandingkan.
AdamW, warmup/cosine, LR classifier 5× backbone, gradient accumulation, AMP, dan EMA disertakan.
Mixup ringan hanya pada fase awal dan tidak dipakai pada fine-tuning akhir.

`best.pt` menyimpan EMA terbaik berdasarkan accuracy, dengan cross-entropy sebagai tie-break.
`last.pt` menyimpan keadaan lengkap untuk resume. Model yang sudah selesai dilewati setelah
config dan fingerprint diperiksa. Jika ingin mengubah hyperparameter, gunakan `RUN_TAG` baru.

Jika CUDA out-of-memory, gunakan run baru dengan batch 4 dan accumulation 8; effective batch
tetap 32, tetapi statistik BatchNorm EfficientNet tetap bergantung pada batch fisik.
''')
markdown('### Pemeriksaan runtime GPU\n\nBF16 hanya dipakai jika didukung native. T4 menggunakan FP16+GradScaler. Probe CPU ditandai skipped.')
code('''
print(json.dumps(runtime_diagnostics(), indent=2))
print(json.dumps(probe_cuda(), indent=2))
''')
code('''
RUN_DIRS = []
for name in MODELS:
    run_name = f'{name}_{RUN_TAG}_seed{CONFIGS[name]["seed"]}'
    run_dir = PROJECT / 'outputs' / run_name
    if ENABLE_TRAINING:
        run_dir = train_model(PROJECT, CONFIGS[name], run_name=run_name,
                              max_hours=MAX_HOURS_PER_MODEL)
    elif not (run_dir / 'best.pt').is_file():
        raise FileNotFoundError(f'Checkpoint belum tersedia: {run_dir / "best.pt"}')
    RUN_DIRS.append(run_dir)
    display(pd.read_csv(run_dir / 'history.csv').tail())
    print(json.dumps(json.loads((run_dir / 'summary.json').read_text()), indent=2))
''')
markdown('''
### 5. Evaluasi akhir, TTA, dan submission masing-masing model

Bandingkan satu resolusi dengan rata-rata prediksi 384+448, tanpa flip teks.
TTA hanya dipilih bila validation accuracy meningkat, atau accuracy sama dan CE lebih rendah.
Peningkatan dari pemilihan tersebut merupakan **skor seleksi pada holdout**, bukan evaluasi test independen.

Laporan mencakup confusion matrix, metrik per kelas, kesalahan ber-confidence tinggi,
serta clean-training-sample accuracy untuk membantu menilai overfitting.
''')
code('''
EVALUATION_DIRS = []
for run_dir in RUN_DIRS:
    evaluation = evaluate_model(PROJECT, run_dir)
    EVALUATION_DIRS.append(evaluation)
    print(run_dir.name)
    print(json.dumps(json.loads((evaluation / 'metrics.json').read_text()), indent=2))
    display(pd.read_csv(evaluation / 'classification_report.csv').head(EXPECTED_CLASSES))
    display(DisplayImage(filename=str(evaluation / 'training_curves.png')))
    display(DisplayImage(filename=str(evaluation / 'confusion_matrix.png')))
''')
markdown('''
### 6. Bandingkan model tunggal dan ensemble 50:50

Tidak ada pencarian ratusan bobot ensemble pada validasi. Dua model tunggal dan satu
ensemble dengan bobot tetap dibandingkan; validation accuracy tertinggi dipilih,
dan log-loss menjadi tie-break. Jika hanya satu model dijalankan, model itu dipakai.
Prediksi selalu diselaraskan ke urutan ID `sample_submission.csv`.
''')
code('''
comparison = compare_models(PROJECT, EVALUATION_DIRS)
display(comparison)
final_submission = pd.read_csv(PROJECT / 'submission.csv', dtype={'id': str})
assert list(final_submission.columns) == list(sample_submission.columns)
assert list(final_submission.id) == list(sample_submission.id)
assert final_submission.shape == sample_submission.shape
assert final_submission.label.between(0, EXPECTED_CLASSES-1).all()
assert not final_submission.isna().any().any()
display(final_submission.head())
print('Submission valid:', final_submission.shape)
''')
markdown('''
### 6b. Submission geometric mean 50:50

Tahap tambahan ini memakai probabilitas kedua CNN, menyelaraskan ID, lalu
menghasilkan `submission_geometric.csv` secara terpisah. CSV dari tahap 6 tetap
tersedia. Tahap ini memerlukan kedua arsitektur dengan 16 kelas kompetisi.
Metrik pada run ini dihitung ulang; skor Kaggle historis 0,88582 tidak disalin.
''')
code('''
GEOMETRIC_SUBMISSION = None
evaluation_by_model = {
    json.loads((path / 'metadata.json').read_text())['model']: path
    for path in EVALUATION_DIRS
}
if EXPECTED_CLASSES == 16 and {'convnext_tiny', 'efficientnet_v2_s'} <= set(evaluation_by_model):
    import numpy as np
    from docvision_calibration.submission import export_geometric_submission
    from docvision_calibration.data import align
    from docvision_calibration.power import PowerMeanPool
    from docvision_v2.engine import metric_dict

    conv_eval = evaluation_by_model['convnext_tiny']
    eff_eval = evaluation_by_model['efficientnet_v2_s']
    GEOMETRIC_SUBMISSION = PROJECT / 'submission_geometric.csv'
    geometric_manifest = export_geometric_submission(
        conv_eval / 'test.npz', eff_eval / 'test.npz',
        PROJECT / 'data/splits/sample_submission.csv', GEOMETRIC_SUBMISSION,
    )
    aligned_validation = np.stack([
        align(path / 'validation.npz', valid_frame.id, valid_frame.label.to_numpy())
        for path in [conv_eval, eff_eval]
    ], axis=1)
    with torch.inference_mode():
        geometric_validation = PowerMeanPool()(torch.as_tensor(aligned_validation, dtype=torch.float64)).numpy()
    geometric_metrics = metric_dict(valid_frame.label.to_numpy(), geometric_validation)
    geometric_manifest.update(validation_metrics=geometric_metrics,
                              fingerprint=manifest['fingerprint'],
                              validation_scope='shared selection holdout; not a Kaggle score')
    GEOMETRIC_SUBMISSION.with_suffix('.manifest.json').write_text(json.dumps(geometric_manifest, indent=2))
    (PROJECT / 'outputs/comparison/geometric_metrics.json').write_text(json.dumps(geometric_metrics, indent=2))
    display(pd.DataFrame([{'model': 'ensemble_geometric_equal', **geometric_metrics}]))
    print('Geometric submission:', GEOMETRIC_SUBMISSION)
else:
    print('Geometric kompetisi dilewati: memerlukan kedua arsitektur dan 16 kelas.')
''', 'geometric-ensemble')
markdown('''
### 7. File output dan paket review

`submission.csv` adalah pilihan berdasarkan validasi. CSV setiap model juga tersedia.
`submission_geometric.csv` adalah kandidat pooling tambahan dari tahap 6b.
`docvision_v2_review.zip` berisi log, metrik, probabilitas, audit, dan diagnostik;
tidak memasukkan dataset atau checkpoint besar. Simpan versi notebook agar checkpoint
tetap tersedia. Jangan menganggap skor validasi sebagai skor public/private Kaggle.
''')
markdown('### Laporan GitHub dari run ini\n\nHasil agregat ditulis ke results/latest dan bagian hasil run terbaru pada README. Skor Kaggle tidak diisi otomatis.')
code('''
latest_results = update_results(PROJECT, EVALUATION_DIRS)
github_zip = export_github_results(PROJECT)
print('Hasil agregat:', latest_results)
print('Paket GitHub:', github_zip)
''')
code('''
review_zip = export_review(PROJECT)
artifact_root = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else PROJECT
artifacts = [PROJECT / 'submission.csv', PROJECT / 'submission_manifest.json', review_zip, github_zip]
if GEOMETRIC_SUBMISSION is not None:
    artifacts += [GEOMETRIC_SUBMISSION, GEOMETRIC_SUBMISSION.with_suffix('.manifest.json')]
artifacts += sorted((PROJECT / 'outputs/comparison').glob('submission_*.csv'))
for source in artifacts:
    target = artifact_root / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    print(target)
    display(FileLink(str(target)))
print('Jika link tidak terbuka, unduh melalui panel Output/Files Kaggle.')
''')
markdown('''
### 8. Pengujian opsional

Pengujian sintetis memeriksa akses TIFF/ZIP, cache, duplikat, resume, ID submission,
gradient accumulation, serta forward/backward backbone asli. Hasilnya bukan skor kompetisi.
Jalankan hanya bila diperlukan; pengujian dipaksa ke CPU dalam proses terpisah.
''')
code('''
if RUN_SMOKE_TESTS:
    import subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    subprocess.run([sys.executable, str(PROJECT / 'tests/smoke_test.py'),
                    str(PROJECT / 'LOCAL_SMOKE_REPORT.json')], env=env, check=True)
else:
    print('Smoke test opsional dilewati. Lihat validation/ pada repository untuk hasil pengembangan.')
''')
markdown('''
### 9. Menafsirkan hasil

Lihat learning curves, confusion matrix, metrik per kelas, dan clean-train sample
bersama-sama. Accuracy selama mixup tidak sama dengan clean-train accuracy.
Pemilihan epoch/TTA/model pada holdout yang sama menambah selection bias.
Hash piksel hanya memisahkan exact duplicate, bukan seluruh near-duplicate.
Untuk eksperimen lanjutan, ubah satu komponen pada split yang sama dengan RUN_TAG
baru dan simpan semua hasil, termasuk percobaan yang tidak membaik.

Notebook arsip v2_gpu_fix menyimpan log training historis. Notebook v2.2 ini
memakai source yang didokumentasikan dan diperiksa ulang. Tanpa checkpoint/CSV
asli, menjalankan ulang training tidak menjamin prediksi identik bit demi bit.
''')
markdown('### Lampiran — analisis dan panduan lengkap\n\n' + (ROOT / 'README.md').read_text())

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
    'language_info': {'name':'python','version':'3.12'},
    'docvision': {'version':'2.2.1','competition_training_executed':False,
                 'refinement':'geometric_mean_50_50',
                 'refinement_validation_report':'validation/refinement_20261028/notebook_geometric_check.json',
                 'validation_report':'validation/notebook_smoke.json'},
}
for index, cell in enumerate(notebook.cells):
    cell.id = hashlib.sha256((str(index) + cell.cell_type + cell.source).encode()).hexdigest()[:16]
nbf.validate(notebook)
path = DEST / 'Document_Classification_v2_2_Kaggle.ipynb'
serialized = nbf.writes(notebook)
if '--check' in sys.argv:
    if not path.is_file() or path.read_text() != serialized:
        raise SystemExit('Notebook is stale: run python scripts/build_notebook.py')
    print('Notebook source synchronization: passed')
else:
    path.write_text(serialized)
    print(path, len(cells), 'cells', path.stat().st_size, 'bytes')
