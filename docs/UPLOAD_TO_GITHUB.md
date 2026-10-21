# Memperbarui hasil setelah run Kaggle

Source dan notebook utama berada pada repository ini. Setelah run baru selesai,
sel pelaporan menghasilkan github_results.zip: README dan results/latest yang
memuat ringkasan, kurva, confusion matrix serta konfigurasi run tersebut.

Ekstrak paket ke checkout repository, tinjau diff, lalu jalankan validator dan
pengujian sebelum commit. Jangan menambahkan outputs/, dataset, kredensial,
best.pt atau last.pt. Source notebook dibuat ulang menggunakan
`python scripts/build_notebook.py` bila dokumentasi/source berubah. Arsip run
v2_gpu_fix dipertahankan sebagai bukti historis; jangan mengganti outputnya
dengan angka dari eksperimen lain.

Untuk merekam skor Kaggle baru, catat nama/ID submission, jenis leaderboard,
tanggal, skor dan hash CSV yang benar-benar disubmit. Skor validasi dan Kaggle
memiliki sumber berbeda dan tidak boleh digabungkan.
