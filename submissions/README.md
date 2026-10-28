# Riwayat submission

Submission terbaru: **[submission_geometric_088582.csv](submission_geometric_088582.csv)**.
Skor Kaggle **0,88582** dilaporkan oleh pemilik repository pada 28 Oktober 2026.
Jenis leaderboard dan peringkat belum diberikan; skor tidak diverifikasi melalui
API Kaggle. Angka ini berbeda dari validation accuracy yang dihitung lokal.

| Berkas | Metode | Validasi / 6.392 | Skor Kaggle dilaporkan |
|---|---|---:|---:|
| [submission_ensemble_equal_088456.csv](submission_ensemble_equal_088456.csv) | Arithmetic mean 50:50 | 0,884230 | 0,88456 |
| [submission_geometric_088582.csv](submission_geometric_088582.csv) | Geometric mean 50:50 | 0,885325 | **0,88582** |

Kedua berkas memuat 7.994 baris, kolom `id,label`, ID unik dengan urutan yang sama,
dan label integer 0–15. Kandidat terbaru mengubah 97 label test. Selisih skor
Kaggle yang dilaporkan adalah 0,00126 atau 0,126 poin persentase. Target 0,90
belum tercapai.

Pemetaan skor terbaru mengikuti berkas utama `submission.csv` pada pembaruan
sebelumnya, yaitu geometric mean 50:50. Kandidat harmonic mean eksploratif
tidak diberi skor Kaggle ini.

[manifest.json](manifest.json) mencatat hash SHA-256 kedua CSV. Identitas
baseline diverifikasi dari arsip probabilitas asli; seluruh file lama di
`results/v2_gpu_fix/` tetap dipertahankan.

```bash
python scripts/verify_submissions.py
```

Perintah tersebut memverifikasi format, ID, hash, dan jumlah perubahan label.
Reproduksi prediksi dijelaskan pada [pipeline](../docs/PIPELINE_FLOW.md).
