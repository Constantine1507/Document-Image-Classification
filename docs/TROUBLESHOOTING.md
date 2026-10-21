# Pemecahan masalah

## FIND was unable to find an engine

Pesan tersebut menunjukkan kegagalan eksekusi convolution CUDA/cuDNN; traceback
saja tidak menentukan satu penyebab pasti. Pada notebook arsip, run berhasil
setelah BF16 native diperiksa, FP16+GradScaler digunakan untuk T4, format tensor
diganti contiguous NCHW dan cuDNN benchmark dinonaktifkan. Source v2.2 menyatukan
perubahan itu dalam engine training dan inference. Jangan menyimpulkan semua
kasus FIND memiliki penyebab yang sama.

Jalankan `python scripts/run.py diagnose` atau sel diagnostik notebook. Pada
GPU, probe kecil mencoba convolution stem dan depthwise beserta backward.
Pada CPU, probe ditandai skipped. Jika tetap gagal, simpan log versi torch,
torchvision, CUDA, cuDNN dan GPU, restart sesi Kaggle dan gunakan pasangan paket
bawaan. Probe berhasil tidak membuktikan resolusi/batch besar pasti muat.

## CUDA out of memory

Gunakan RUN_TAG baru dan batch 4 dengan accumulation_steps 8. Effective batch
tetap 32, tetapi statistik BatchNorm EfficientNet mengikuti batch fisik.
Turunkan eval_batch_size terpisah jika kegagalan terjadi saat evaluasi.

## RUN_DIRS atau compare_models belum didefinisikan

Jalankan notebook dari sel setup/source/import sampai training secara berurutan.
Sel perbandingan memerlukan EVALUATION_DIRS yang berisi evaluasi nyata. Jangan
mengisi daftar kosong untuk menutupi checkpoint yang hilang. Jika sesi restart,
buat ulang import dan pasang output Kaggle tersimpan yang berisi best.pt/last.pt.

## Resume ditolak

Resume memeriksa konfigurasi, split/data dan hash source. Perubahan source
termasuk docstring dapat mengubah hash. Gunakan source arsip yang sesuai untuk
melanjutkan run historis, atau RUN_TAG baru untuk source v2.2. Jangan mengedit
hash checkpoint agar pemeriksaan lolos. Inferensi checkpoint dapat dilakukan
setelah fingerprint data diverifikasi; checkpoint wajib berasal dari sumber
yang dipercaya dan arsitektur/config yang sesuai.

## Dataset tidak ditemukan

DATA_ROOT harus menunjuk folder train.csv, test.csv dan sample_submission.csv
kompetisi gambar. Dataset Structural Damage adalah tugas berbeda. Bila gambar
berada terpisah, tambahkan IMAGE_ROOTS. Gambar/ZIP yang sudah ada dibaca langsung.
Dataset sumber tetap diperlukan ketika menggunakan checkpoint.

## Output hilang setelah sesi Kaggle

Simpan versi notebook beserta output sebelum sesi ditutup. Path sesi lama tidak
berarti berkas masih tersedia. Tanpa best.pt/last.pt, model historis tidak bisa
dipulihkan dari history.csv atau notebook yang hanya memuat log training.
