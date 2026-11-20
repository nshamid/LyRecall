# LyRecall

Pencarian lagu dari potongan lirik yang tidak lengkap atau salah dengar, memakai *hybrid retrieval*
(BM25 + TF-IDF *n*-gram karakter + *embedding* Sentence-Transformer, digabung dengan Reciprocal Rank Fusion).

Proyek mata kuliah Temu Kembali Informasi, Tahun Ajaran 2026-2027.
Laporan: *Hybrid Retrieval untuk Pencarian Lagu dari Potongan Lirik yang Tidak Lengkap dan Salah Dengar*.

**Demo:** https://lyrecall.vercel.app

(server pencarian berjalan di laptop, jadi demo hanya aktif saat server menyala)

**Kode antarmuka web:** https://github.com/indrinanc/lyrecall-web

## Ringkasan

- **Masukan:** satu potongan lirik (bahasa Inggris). **Keluaran:** 10 lagu teratas beserta cuplikan, skor, dan label kecocokan (Exact / Fuzzy / Semantic).
- **Data:** 99.821 lagu dari *genius-lyrics-cleaned* dipecah menjadi 1.843.304 *chunk* (jendela 3 baris, langkah 2).
- **Metode:** tiga *retriever* digabung dengan RRF di tingkat *chunk* (k = 3, kedalaman 100, bobot 1:1:1), lalu skor maksimum per lagu.
- **Evaluasi:** 2.981 kueri uji sintetis *known-item*: exact, parsial, berisik, dan parafrase.

## Hasil utama (MRR pada 2.981 kueri uji)

| Sistem | Exact | Parsial | Berisik | Parafrase |
|---|---|---|---|---|
| BM25 | 0,988 | 0,957 | 0,936 | 0,142 |
| TF-IDF karakter | 0,979 | 0,892 | 0,951 | 0,092 |
| MiniLM (dense) | 0,907 | 0,760 | 0,749 | 0,221 |
| RRF leksikal (BM25 + TF-IDF) | 0,989 | 0,947 | 0,962 | 0,136 |
| **Hybrid 3 *retriever* (sistem akhir)** | 0,986 | 0,939 | 0,955 | 0,202 |

Menambahkan *retriever* semantik menaikkan MRR parafrase dari 0,136 menjadi 0,202, dengan biaya
penurunan 0,004–0,008 pada tiga jenis kueri lain. Rincian lengkap ada di laporan.

## Isi repositori

Notebook dijalankan di Kaggle, berurutan sebagai berikut.

| Notebook | Fungsi |
|---|---|
| `audit-dataset.ipynb` | Audit dataset awal |
| `pembersihan.ipynb` | Pembersihan dan penyaringan lirik |
| `chunking.ipynb` | Pemecahan lirik menjadi *chunk* |
| `indeks-leksikal.ipynb` | Indeks BM25 dan TF-IDF karakter |
| `embedding.ipynb` | *Embedding* MiniLM (GPU) |
| `kueri-sintetis.ipynb` | Kueri exact, parsial, dan berisik |
| `rff-agregasi-dan-evaluasi.ipynb` | Fusi RRF, agregasi ke lagu, dan evaluasi awal |
| `kueri-parafrase.ipynb` | Kueri parafrase (Qwen2.5-3B-Instruct) |
| `evaluasi-dengan-kueri-parafrase.ipynb` | Evaluasi akhir: penyetelan RRF, bobot, dan uji signifikansi |
| `backend/` | Server pencarian (FastAPI) |

## Cara menjalankan ulang

1. Jalankan notebook di Kaggle sesuai urutan tabel di atas (GPU T4 untuk *embedding* dan parafrase).
2. Setiap notebook memakai keluaran notebook sebelumnya sebagai input (lihat bagian "Input" di Kaggle).
3. Artefak indeks (±5,4 GB) tidak disertakan di repositori karena ukurannya.

## Menjalankan server pencarian

Memerlukan sekitar 9–10 GB RAM dan ±12 GB ruang disk untuk artefak indeks.

```bash
cd backend
pip install -r requirements.txt
# atur DATA_REPO dan HF_TOKEN sebagai variabel lingkungan; jangan menulis token di kode
uvicorn app:app --port 8000
```

Antarmuka web (halaman statis dan fungsi Vercel yang membaca `API_URL`) ada di
[indrinanc/lyrecall-web](https://github.com/indrinanc/lyrecall-web). Server diekspos dengan Cloudflare Tunnel.

## Data dan model

- Dataset: [theelderemo/genius-lyrics-cleaned](https://huggingface.co/datasets/theelderemo/genius-lyrics-cleaned)
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Parafrase kueri: `Qwen2.5-3B-Instruct`
- Pelafalan kata untuk kueri berisik: CMUdict

Lirik berhak cipta milik pemilik aslinya. Repositori ini tidak menyertakan teks lirik, dan antarmuka hanya menampilkan cuplikan pendek.

## Keterbatasan

Kueri sintetis (belum divalidasi dengan pengguna nyata); mutu parafrase tidak dinilai manusia;
subset sekitar 100 ribu lagu; ukuran *chunk* dan *stop-word* tidak diablasi; latensi diukur di Kaggle.

## Tim

| Nama | NIM | Kontribusi |
|---|---|---|
| Nabilah Shamid | 09021382328147 | ‹sesuai laporan› |
| Indrina Nur Chairunnisya | 09021382328157 | ‹sesuai laporan› |
| Shalaisya Fattiha Ramadhani | 09021382328161 | ‹sesuai laporan› |
| Raka Athallah Ananta | 09021382328163 | ‹sesuai laporan› |

## Lisensi

‹pilih lisensi, mis. MIT untuk kode›. Dataset dan model mengikuti lisensi masing-masing.
