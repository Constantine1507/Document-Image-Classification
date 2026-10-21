# Kesesuaian Tugas Supervised Machine Learning 2026

Pemetaan ini mengikuti dokumen tugas yang diberikan, bukan rubrik tugas tabular sebelumnya.

| Ketentuan | Implementasi/bukti |
|---|---|
| Neural network menggunakan PyTorch | torch/torchvision; ConvNeXt Tiny dan EfficientNetV2-S |
| Pretrained dan fine-tuning diperbolehkan | Bobot awal ImageNet-1K resmi torchvision; semua parameter di-fine-tune |
| Struktur folder/script | src, scripts, configs, notebooks, tests, results, docs |
| Docstring setiap function/class .py | Diperiksa oleh scripts/validate_repository.py |
| .gitignore data/ dan checkpoint/ | Kedua direktori dicantumkan, juga outputs/ dan berkas bobot |
| Tidak mengunggah dataset besar >100 MB | Dataset/weights tidak dibundel; validator memeriksa ukuran berkas |
| README cara unduh dan menjalankan | Tautan data/undangan, panduan Kaggle dan terminal |
| Interpretasi hasil | Perbandingan aktual, kurva, kelas sulit, overfitting dan batas bukti |
| Notebook hasil eksekusi | notebooks/executed/v2_gpu_fix_kaggle.ipynb asli, hash diverifikasi |
| Workflow mengikuti contoh praktikum | Alur audit–split–training–evaluasi–submission tersedia; kecocokan persis contoh praktikum belum dapat diverifikasi karena contoh tidak disertakan |

Dokumen mencantumkan bobot GitHub 70%, public leaderboard 10%, private 20%;
deadline 4 November 2026 pukul 22.00 WIB dan maksimal 5 submission Kaggle per hari.
Pengerjaan bersifat individu. Tidak ditemukan larangan eksplisit ensemble dalam
teks tugas yang diberikan. Penggunaan pandas, NumPy, Pillow, matplotlib dan
scikit-learn untuk data/metrik tidak mengubah framework training yang tetap PyTorch.
TensorFlow dan JAX tidak digunakan. Pipeline/ColumnTransformer bukan syarat
yang tercantum pada dokumen tugas gambar ini.

Nilai akhir, ranking dan penerimaan dosen tetap ditentukan penilaian sebenarnya.
Sebelum pengumpulan, pastikan repository public dan pilih submission Kaggle
yang benar. Repository mencatat skor Kaggle 0,88456 sebagai laporan pemilik,
bukan sebagai verifikasi leaderboard atau klaim ranking.
