# Metodologi dan batasan

## Data langsung dan format yang didukung

CSV sumber tidak dimodifikasi. Kolom path dapat bernama `filename` atau
`image_path` (juga `path`, `filepath`, `file_path`). Label kompetisi ini harus
integer 0 sampai 15. Submission mengikuti template `id,label` dan disejajarkan
berdasarkan ID, termasuk ID yang memiliki nol di depan.

Jika gambar belum diekstrak dan hanya tersedia di ZIP, anggota ZIP dibaca
langsung tanpa ekstraksi. Ini menghemat ruang penyimpanan tetapi dapat
memperlambat pembacaan TIFF pada setiap epoch. Jika file gambar sudah ada,
file tersebut diprioritaskan. Tidak ada konversi massal atau salinan otomatis.
Pembacaan dan decoding minibatch tetap diperlukan oleh model.

## Audit dan pembagian validasi

- Gambar training tidak terbaca dilaporkan dan dikeluarkan lewat metadata.
  Lebih dari 1% kegagalan (minimal ambang dua file) menghentikan proses agar
  salah path tidak menyebabkan penghapusan ribuan baris secara diam-diam.
- Gambar test gagal dibaca selalu menghentikan proses. Tidak ada prediksi
  menggunakan gambar kosong atau penghapusan baris submission.
- Semua baris kelompok piksel identik dengan label berbeda dikarantina.
  Label tidak diubah menggunakan voting atau asumsi nama folder.
- Duplikat berlabel sama ditempatkan dalam sisi split yang sama. Selain
  akurasi per baris, laporan menyertakan akurasi per gambar unik.
- Train/test exact overlap dilaporkan. Tidak ada penyalinan label ke test,
  lookup ID, atau penggunaan nama folder/filename sebagai fitur model.
- Split lama dapat dipakai kembali hanya jika ID, path, label, cakupan data,
  dan pemisahan kelompok duplikat lolos verifikasi. Jika tidak, notebook
  membuat holdout baru sekitar 80:20 dan mencatat asal split.
- Hash persis tidak mendeteksi near-duplicate atau halaman terkait dari satu
  dokumen. Bila identitas sumber dokumen tersedia, perlu split berdasarkan
  identitas tersebut sebelum menginterpretasikan hasil sebagai generalisasi
  ke sumber dokumen yang benar-benar baru.

## Resep training dan alasannya

| Komponen | Default | Tujuan dan batasan |
|---|---|---|
| Bobot awal | ImageNet-1K torchvision | Transfer fitur visual umum; tidak memakai checkpoint RVL-CDIP yang bisa tumpang tindih dengan dataset |
| Backbone | ConvNeXt Tiny; EfficientNetV2-S | Dua arsitektur untuk dibandingkan dan diuji ensemble, bukan jaminan satu selalu unggul |
| Resolusi | 320, lalu 384 mulai epoch 15 | Memberi kesempatan model belajar detail halaman setelah adaptasi awal |
| Epoch | Maksimum 20 per model | Titik awal eksperimen; checkpoint dipilih dari validasi |
| Batch | 8 × akumulasi 4 = efektif 32 | Membatasi kebutuhan VRAM; akumulasi tidak membuat statistik BatchNorm setara dengan batch fisik 32 |
| Learning rate | ConvNeXt 1e-4; EfficientNet 2e-4 | LR backbone konservatif, classifier 5× lebih besar |
| Scheduler | Warmup 2 epoch, cosine decay | Mengurangi perubahan mendadak pada awal fine-tuning |
| Optimizer | AdamW, decay 0,01 | Bias dan parameter norm tidak dikenai weight decay |
| Augmentasi | Affine ringan, brightness/contrast ringan | Menjaga sebagian besar layout; tidak ada horizontal flip atau random crop besar |
| Label smoothing | 0,05, lalu 0,02 | Regularisasi ringan; tetap perlu diuji terhadap validasi |
| Mixup | alpha 0,1; peluang 0,25 pada fase awal | Dapat membantu regularisasi; dimatikan saat fine-tuning resolusi tinggi; manfaatnya bukan kepastian |
| EMA | decay maksimal 0,999 dengan warmup | Validasi dan best checkpoint memakai EMA; last checkpoint juga menyimpan bobot mentah untuk resume |
| AMP | BF16 bila didukung, selain itu FP16 + scaler | Efisiensi GPU; CPU tests memakai FP32 |
| TTA | Bandingkan 384 saja versus rata-rata 384+448 | Dipakai hanya jika membantu validation accuracy, dengan CE sebagai tie-break |
| Ensemble | Rata-rata probabilitas 50:50 | Dibandingkan dengan dua model tunggal; ensemble yang lebih buruk tidak dipilih |

Resolusi lebih tinggi dan TTA menambah waktu. Training menggunakan satu GPU
yang terlihat sebagai `cuda:0`; GPU kedua tidak otomatis mempercepat training.
Tidak ada klaim waktu selesai karena GPU, I/O, dan kompresi TIFF memengaruhinya.

## Mengukur kualitas secara jujur

`best_validation_accuracy` adalah skor holdout untuk memilih epoch/model,
bukan skor public/private Kaggle. Pemilihan epoch, TTA, dan model pada holdout
yang sama dapat membuat angka terpilih optimistis. Jangan mencoba banyak seed
split lalu hanya melaporkan yang tertinggi.

Periksa secara bersama-sama:

1. Kurva validation accuracy, macro-F1 dan loss.
2. Akurasi clean training sample dengan preprocessing evaluasi. Train accuracy
   selama mixup bukan pembanding langsung karena target dan augmentasinya berbeda.
3. Recall per kelas dan confusion matrix, terutama kelas yang bentuknya mirip.
4. Kesalahan dengan confidence tinggi: gunakan untuk review label/layout,
   bukan otomatis mengganti label agar skor validasi naik.
5. Jumlah contoh yang benar-benar berubah ketika TTA/ensemble memperbaiki skor.
   Perbedaan yang hanya beberapa contoh belum membuktikan generalisasi lebih baik.
6. Untuk laporan final yang lebih kuat, gunakan holdout lain yang tidak dipakai
   memilih konfigurasi atau cross-validation berbasis kelompok dengan budget memadai.

Tidak ada inverse-frequency loss atau oversampling secara default karena
target utama adalah accuracy. Jika minoritas memiliki recall buruk, uji satu
perubahan pada split yang tetap dan bandingkan accuracy serta macro-F1.

## Strategi peningkatan sesudah run nyata

- Jika training dan validasi masih meningkat sampai akhir, tambah horizon di
  **run baru**. Jangan mengubah horizon scheduler pada resume eksperimen lama.
- Jika clean-train tinggi dan validasi stagnan/turun, coba kurangi epoch atau
  LR, dan bandingkan regularisasi satu per satu.
- Jika training juga rendah, periksa mapping label, preview gambar, bobot
  pretrained, LR, dan augmentasi sebelum memperbesar model.
- Jika fitur teks kecil sering gagal, bandingkan fine-tuning 448 pada satu run
  terkontrol dengan batch lebih kecil. Resolusi tinggi bukan obat untuk label salah.
- Jika satu arsitektur jelas lebih baik dan GPU terbatas, prioritaskan model
  tersebut. Ensemble hanya berguna bila kesalahannya saling melengkapi.
- Refit seluruh train belum otomatis dilakukan dalam v2: submission default
  berasal dari model yang mempunyai bukti validasi. Setelah konfigurasi dan
  horizon terkunci, refit dapat menjadi eksperimen terpisah tanpa klaim memiliki
  validation score independen.

## Checkpoint dan reproduksi

- `best.pt`: bobot EMA terpilih, config, epoch, kelas, metrik, fingerprint data,
  dan hash source; digunakan untuk inferensi tanpa download pretrained ulang.
- `last.pt`: bobot mentah dan EMA, optimizer, scaler, RNG, langkah LR, history,
  serta status selesai; digunakan untuk melanjutkan epoch berikutnya.
- Resume menolak config, data/split, atau source yang berubah. Gunakan `RUN_TAG`
  baru untuk eksperimen baru. Memakai checkpoint model lain akan ditolak.
- Semua file training, termasuk checkpoint, tetap di output Kaggle; paket
  review tidak memasukkan checkpoint besar atau dataset.
- Fingerprint split mencakup ID, label, hash piksel, ID validasi, dan urutan
  template. Prediksi kedua model harus memiliki fingerprint dan urutan sama
  sebelum ensemble boleh dibuat.
- Cache audit menggunakan ukuran/mtime file atau CRC ZIP; cache diasumsikan
  untuk dataset yang tidak dimodifikasi secara tersembunyi. Hapus cache bila
  file sengaja diubah sambil mempertahankan seluruh metadata tersebut.
- Seed disimpan, tetapi CUDA tidak dijanjikan identik bit demi bit antarmesin.


## Pemilihan AMP

Source v2.2 hanya memilih BF16 jika GPU mendukungnya secara native. GPU T4 memakai FP16 dengan GradScaler; CPU memakai FP32. Autocast dan GradScaler menggunakan kebijakan yang sama. Format tensor contiguous NCHW dan cuDNN benchmark nonaktif mempertahankan perbaikan yang dipakai pada run GPU.

Referensi: [PyTorch AMP](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html), [ConvNeXt Tiny](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.convnext_tiny.html), [EfficientNetV2-S](https://docs.pytorch.org/vision/0.23/models/generated/torchvision.models.efficientnet_v2_s.html).
