# Hasil tiga putaran iterasi submission RVL-CDIP

27 Oktober 2026 — seluruh nilai pada laporan ini dihitung dari probabilitas
yang benar-benar disertakan dalam `docvision_v2_review.zip`.

## Hasil akhir

Kandidat utama adalah **geometric mean 50:50** dari dua CNN. Akurasi validasinya
**0,885325**, naik dari **0,884230**. Ada 27 kesalahan yang diperbaiki dan 20
prediksi benar yang menjadi salah: tambahan benar bersih **7 dari 6.392**.
Kandidat ini mengubah **97 dari 7.994 label test**. Perubahan label test belum
dapat dinyatakan benar sebelum dievaluasi Kaggle.

**Target >0,90 belum tercapai.** Skor Kaggle kandidat baru belum tersedia.
Angka 0,88456 adalah skor baseline yang dilaporkan pengguna, bukan nilai hasil
training/kalibrasi baru. Tidak ada label test yang digunakan untuk optimasi.

| Metode atau prosedur | Akurasi validasi/koreksi | Benar / 6.392 | Jenis penilaian |
|---|---:|---:|---|
| Baseline arithmetic mean 50:50 | 0,884230 | 5.652 | Holdout asli |
| Kandidat utama geometric mean 50:50 | **0,885325** | **5.659** | Aturan tetap pada holdout; dipilih setelah perbandingan eksploratif |
| Putaran 1: pemilihan korektor | 0,882979 | 5.644 | Nested group CV lapisan koreksi |
| Putaran 2: korektor fokus prediksi ambigu | 0,883605 | 5.648 | Nested group CV lapisan koreksi |
| Putaran 3: optimasi pooling global | 0,885169 | 5.658 | Parameter dipilih di training partition tiap fold |
| Alternatif eksploratif: harmonic mean 0,35/0,65 | 0,887359 | 5.672 | **Skor pemilihan parameter pada seluruh holdout**, bukan skor cross-fit |

Nilai terakhir tidak dapat dibandingkan begitu saja dengan nilai cross-fit.
Parameter 0,35/0,65 dipilih dari 85 kombinasi menggunakan holdout yang sama.
Estimasi prosedur pencariannya pada baris yang tidak digunakan fitting lebih
rendah, yaitu 0,885169. File alternatif tetap disertakan sebagai kandidat
eksploratif untuk dinilai Kaggle, dengan risiko overfitting pemilihan parameter.

## Formula dan isi berkas

Kandidat utama memakai:

\[
p_{\mathrm{baru}}(c\mid x)=
\frac{\sqrt{p_{\mathrm{ConvNeXt}}(c\mid x)\,p_{\mathrm{EfficientNet}}(c\mid x)}}
{\sum_k\sqrt{p_{\mathrm{ConvNeXt}}(k\mid x)\,p_{\mathrm{EfficientNet}}(k\mid x)}}.
\]

Label akhir adalah argmax. Seluruh prediksi masih berasal dari dua neural
network PyTorch; pooling diimplementasikan sebagai `torch.nn.Module`.
Formula alternatif adalah weighted harmonic mean:

\[
q(c\mid x)\propto
\left(\frac{0.35}{p_{\mathrm{ConvNeXt}}(c\mid x)}+
\frac{0.65}{p_{\mathrm{EfficientNet}}(c\mid x)}\right)^{-1}.
\]

Alternatif mengubah **145 label test**. Pada data pemilihan parameter,
alternatif memperbaiki 43 kesalahan dan merusak 23 prediksi benar, sehingga
kenaikan bersih 20. Nilai itu tetap mengandung selection bias.

Kedua CSV memiliki tepat 7.994 baris, kolom `id,label`, ID unik yang selaras
dengan arsip asal, dan label integer 0–15. Tidak ada baris dihapus atau ditebak
karena tidak tersedia. File utama berbeda dari baseline secara byte dan label.

| File | SHA-256 |
|---|---|
| Baseline | `854d620cf37cfe0ac23106739eac3f6bb41f24ce18b14d7fbd4dc1c5e3025a76` |
| submission.csv | `812d1b43a69b2cfc5875f8930a38db54b6e33c2db259dc4252a873882d35afa1` |
| submission_exploratory_power.csv | `85340302eb9b82aa32641df60ef56ca2f679e3986057da7d0f988f8c259cf3eb` |

## Audit sumber

Arsip memuat probabilitas lengkap untuk 6.392 gambar validasi dan 7.994 test,
label validasi, kelompok hash piksel, konfigurasi serta metadata checkpoint.
Probabilitas dapat mereproduksi seluruh label CSV baseline yang sebelumnya
dikirim pengguna, termasuk hash berkasnya. Akurasi baseline 5.652/6.392 juga
cocok persis. Dua baris validasi berbagi hash dengan baris lain; pengelompokan
menghasilkan 6.390 kelompok unik.

Checkpoint asal: ConvNeXt Tiny EMA epoch 18 dan EfficientNetV2-S EMA epoch 16.
Kedua backbone memakai inferensi 384 piksel. Data gambar dan bobot CNN tidak
ada dalam paket, sehingga tidak ada training ulang backbone pada pekerjaan ini.
Yang benar-benar dilatih adalah lapisan koreksi kecil pada probabilitas CNN.

Data audit awal tetap 31.959 training bersih: 25.567 train dan 6.392 validasi,
setelah satu gambar rusak serta 43 baris konflik label dikarantina. Tidak ada
penghapusan, penggantian, atau relabeling tambahan selama iterasi ini.

## Diagnosis yang kini dapat diverifikasi

- Kedua CNN mempunyai top-1 salah pada **571** gambar validasi. Mereka setuju
  pada 5.727 gambar; **378** kesepakatan itu sama-sama salah.
- Pada 665 gambar ketika CNN berbeda pendapat, ensemble awal benar pada 303.
  ConvNeXt sendiri benar ketika EfficientNet salah pada 240 gambar; arah
  sebaliknya 232 gambar. Tidak ada satu model yang dominan pada semua kasus.
- Oracle yang selalu memilih top-1 CNN yang benar akan mencapai **0,910670**.
  Oracle memerlukan informasi jawaban yang tidak tersedia pada inferensi;
  angkanya merupakan batas diagnostik untuk pemilihan top-1, bukan hasil model.
- Coverage dua kelas tertinggi ensemble adalah **0,936640**. Mengetahui bahwa
  jawaban ada pada dua calon masih berbeda dengan mengetahui calon yang benar.
- Ada **229 kesalahan** dengan confidence ensemble minimal 0,90. Pada preview,
  beberapa halaman hampir kosong atau memiliki layout yang beririsan antar-
  kategori. Ini mendukung pemeriksaan kesalahan lebih lanjut, bukan izin
  mengubah label ground truth menurut dugaan visual.

Diagnosis overfitting lebih kuat setelah review tersedia: pada sampel clean
train 2.048 gambar, ConvNeXt mencapai 100% dan EfficientNet 99,902%, sedangkan
validasi masing-masing 87,437% dan 87,312%. Ini menggunakan evaluasi single-scale
yang sebanding, meskipun sampel train belum mewakili semua training images.

TTA 384+448 memang menurunkan akurasi: ConvNeXt 87,437% → 86,859%, EfficientNet
87,312% → 86,984%. Karena itu, tidak ada aktivasi ulang TTA secara otomatis.

## Apa yang dijalankan

**Putaran 1 — 15 konfigurasi.** Dua model tunggal, arithmetic/geometric mean,
temperature dan bobot, tambahan bias kelas, bobot per kelas, residual linear
dengan beberapa regularisasi, blending residual, MLP residual, serta optimasi
bias keputusan. Pemilihan dilakukan di tiga inner folds pada setiap satu dari
lima outer folds. Hasil outer aggregate 88,298%: 11 kesalahan diperbaiki tetapi
19 prediksi benar rusak. Gate promosi gagal, sehingga output otomatis putaran
ini mempertahankan baseline.

**Putaran 2 — 10 konfigurasi.** Menambah pembobotan contoh ambigu, loss halus
yang lebih dekat ke classification error, dan neural selector khusus ketika
dua CNN berbeda pendapat. Lima outer folds baru dengan tiga inner folds tetap
digunakan. Hasil outer aggregate 88,360%: 13 kesalahan diperbaiki dan 17 prediksi
benar rusak. Gate promosi kembali gagal. Hasil gagal tetap disimpan.

**Putaran 3 — 85 kombinasi global.** Bobot ConvNeXt dibatasi 0,30–0,70, dengan
lima eksponen pooling. Setiap fold memilih parameter menggunakan training
partition saja. Dua prosedur diuji: pemenang training tunggal serta rata-rata
kombinasi dalam jarak empat prediksi benar dari optimum training. Keduanya
memberi 88,517% cross-fit; log-loss memburuk dibanding geometric mean tetap.
Skor maksimum pada seluruh data pemilihan parameter lebih tinggi, 88,736%,
tetapi tidak diperlakukan sebagai penilaian independen.

Karena prosedur terlatih tidak memberi peningkatan stabil, kandidat utama
memakai geometric mean tetap yang sederhana. Ini keputusan akhir eksploratif
setelah membandingkan hasil, bukan hasil kelulusan gate kedua run nested.
Kedua gate tidak diubah agar seolah-olah menerima kandidat yang gagal.

## Kekuatan bukti dan batas target

Kenaikan kandidat utama hanya **0,1095 poin persentase**. Bootstrap berpasangan
berdasarkan kelompok memberi interval deskriptif 95% sekitar **−0,0939 sampai
+0,3129 poin persentase**. Interval mencakup nol; perbedaan belum kuat secara
statistik. Nilai McNemar deskriptif sekitar 0,382. Untuk alternatif, interval
tanpa koreksi tampak positif, tetapi terpengaruh pemilihan terbaik dari 85
kombinasi, sehingga tidak menjadi bukti signifikansi yang dapat diandalkan.

Grafik `accuracy_gain_diagnostics.png` memperlihatkan besarnya perubahan dan
interval tersebut. Pemeriksaan NumPy independen mereproduksi seluruh 7.994 label
pada kedua kandidat; selisih probabilitas maksimum terhadap implementasi
PyTorch kurang dari 7 × 10⁻¹⁶. Rincian tersedia pada `REPRODUCTION_CHECK.json`.
Tabel `per_class_final_comparison.csv` memuat precision, recall, F1 dan support
untuk seluruh 16 kelas pada baseline dan kedua kandidat.

Masing-masing outer assessment row tidak dipakai melatih korektor pada fold
itu. Namun base checkpoint sebelumnya dipilih dengan holdout ini, dan rancangan
putaran lanjutan dipengaruhi hasil putaran awal. Semua angka ini merupakan
diagnostik pengembangan; bukan estimasi independen seluruh workflow dan bukan
skor public/private Kaggle. Repeated selection tetap mempunyai risiko overfit.

Untuk melewati 90% pada 6.392 baris, diperlukan minimal 5.753 benar. Kandidat
utama masih membutuhkan **94 tambahan benar bersih**; alternatif pada data
pemilihan parameter masih membutuhkan 81. Memaksakan kuota kelas atau terus
mencari parameter hingga skor tampak tinggi tidak memberikan informasi visual
baru untuk memperbaiki error yang sama-sama dilakukan kedua CNN.

Perbaikan berikutnya yang substansial perlu diuji melalui model/fitur visual
tambahan, fine-tuning resolusi yang sesuai, atau penggunaan seluruh data train
bersih dengan evaluasi yang memadai. Ketiganya memerlukan akses gambar atau
embedding/checkpoint yang relevan; tidak dapat dijalankan ulang dari CSV label
dan dua matriks probabilitas ini saja.

## Penggunaan

Kirim kandidat utama ke Kaggle dan catat skor yang diterima. Alternatif dapat
dinilai sebagai eksperimen terpisah; pertahankan baseline lama untuk perbandingan.
Jangan menulis angka validasi laporan ini sebagai skor Kaggle di README.

Paket source menyertakan skrip reproduksi CSV, konfigurasi tiga putaran, hasil
per fold, hasil per kelas, hash, probabilitas kandidat, dan parameter PyTorch.
Tidak ada TensorFlow/JAX atau model klasifikasi non-neural yang digunakan.
Materi detail minggu 3–4 belum tersedia, sehingga pemetaan persis konvensi
perkuliahan tetap perlu dicocokkan; struktur modular dan bukti eksperimen telah
dipertahankan.
