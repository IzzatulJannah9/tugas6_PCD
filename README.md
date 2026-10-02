# Deteksi Keberadaan Tanda Tangan pada Ijazah (Mini Project Pengolahan Citra Digital)

Pipeline klasik (tanpa deep learning): **crop ROI → grayscale → thresholding (global, Otsu, adaptive) → morfologi (opening + closing) → fitur area → aturan `SIGNATURE PRESENT / ABSENT`**.

Data: 9 varian citra satu ijazah (high-quality, low-contrast, blurred, high-noise, low-res, faded, color-shift, JPEG artifact, combined degradation) dari `IJAZAH_PCD.pdf`.

> **Catatan tentang "kepala sekolah"**: ijazah ini dari universitas, jadi padanannya adalah **Rektor** (kepala institusi). Program juga menguji tanda tangan **Dekan** dan **pemilik ijazah**.

## Struktur

```
.
├── run_pipeline.py        # program utama (batch & satu citra)
├── extract_from_pdf.py    # ekstrak 9 citra dari PDF -> data/certs/
├── src/sigdetect.py       # crop, threshold, morfologi, fitur, aturan keputusan
├── data/IJAZAH_PCD.pdf    # sumber
├── data/certs/*.png       # 9 citra (sudah ditegakkan)
└── outputs/               # hasil: results.csv, metrics.md, gambar perbandingan
```

## Cara menjalankan

Python 3.9+.

```bash
git clone <URL-REPO-ANDA>
cd <nama-repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (opsional) regenerasi citra dari PDF. data/certs/ sudah disertakan.
python extract_from_pdf.py data/IJAZAH_PCD.pdf

# 1) Batch: uji semua citra (positif + negatif), cetak tabel akurasi, simpan gambar ke outputs/
python run_pipeline.py

# 2) Satu citra + satu ROI
python run_pipeline.py --image data/certs/03_blurred.png --roi rektor
python run_pipeline.py --image data/certs/03_blurred.png --roi kosong_kiri
```

Pilihan `--roi`: `rektor`, `dekan`, `pemilik` (ada tanda tangan) dan `kosong_kiri`, `kosong_atas`, `teks_nama`, `teks_nomor` (tidak ada tanda tangan).

Contoh keluaran:

```
[otsu    ] fg_pixels= 1665 fg_ratio=0.057 komponen=1 connectivity=1.00 -> SIGNATURE PRESENT
KEPUTUSAN AKHIR (Otsu + morfologi): SIGNATURE PRESENT
```

File hasil di `outputs/`: `results.csv` (per kasus: jumlah piksel foreground & prediksi tiap metode), `metrics.md`, `threshold_comparison_rektor.png`, `threshold_comparison_dekan.png`, `morphology_effect.png`, `threshold_too_low_high.png`, `fg_ratio_vs_threshold.png`, `decision_examples.png`.

## Metode

1. **Crop ROI**: koordinat relatif (persen lebar/tinggi), jadi berlaku untuk semua resolusi. ROI di-resize ke lebar 300 px agar jumlah piksel dan ukuran kernel konsisten antar varian.
2. **Grayscale** (`cv2.cvtColor`) + Gaussian blur 3×3.
3. **Thresholding** (tinta = foreground, `THRESH_BINARY_INV`):
   - Global tetap, T = 127
   - Otsu (T dihitung otomatis dari histogram)
   - Adaptive Gaussian (blok 31, C = 12)
4. **Morfologi**: opening elips 2×2 (buang bintik noise), lalu closing elips 5×5 (sambung goresan putus).
5. **Fitur**: `fg_pixels`, `fg_ratio`, jumlah komponen, `connectivity` (= komponen terbesar / total foreground), lebar & tinggi bounding box relatif, `ink_contrast` (= (median latar − persentil-1) / median).
6. **Aturan** (`src/sigdetect.py`, `RULE`): **PRESENT** jika semua terpenuhi:
   - `ink_contrast ≥ 0.10`: ada piksel yang jauh lebih gelap dari kertas. Ini mencegah Otsu "mengarang" foreground dari noise/gradien kertas kosong.
   - `0.03 ≤ fg_ratio ≤ 0.40`
   - `connectivity ≥ 0.55`: tanda tangan = satu goresan menyambung; teks tercetak = banyak komponen kecil.
   - `bbox_w ≥ 0.35` dan `bbox_h ≥ 0.45`

## Hasil pengujian

63 kasus uji = 9 varian × (3 ROI bertanda tangan + 4 ROI tanpa tanda tangan).

| Metode threshold | Akurasi | Benar/Total | FP | FN |
|---|---|---|---|---|
| Global (T=127) | 60.3% | 38/63 | 0 | 25 |
| **Otsu** | **100%** | 63/63 | 0 | 0 |
| Adaptive Gaussian | 90.5% | 57/63 | 0 | 6 |

- **Global T=127 gagal** pada 25 kasus. Pada citra terang/low-contrast/blur, goresan tipis punya intensitas > 127 sehingga tanda tangan terpotong-potong atau hilang. Pada citra *faded* (06) yang kertasnya gelap, justru berhasil, tetapi satu angka tetap tidak bisa cocok untuk semua kondisi.
- **Otsu** menyesuaikan T tiap citra (T berkisar ≈ 138–218), sehingga stabil di semua varian.
- **Adaptive** bagus untuk pencahayaan tidak merata, tetapi pada tanda tangan pemilik (kecil dan tipis) sering terlalu sensitif terhadap noise/latar.
- Morfologi (`morphology_effect.png`): opening membuang bintik, closing menyatukan goresan yang putus sehingga jumlah komponen turun dan `connectivity` naik.

## Analisis

**Mengapa thresholding diperlukan sebelum analisis keberadaan tanda tangan?**
Pertanyaan "ada tanda tangan atau tidak" butuh pemisahan piksel tinta (objek) dari kertas (latar). Citra grayscale hanya berisi 256 tingkat keabuan; kita tidak bisa menghitung luas, komponen, atau bentuk goresan tanpa label biner foreground/background. Thresholding mengubah masalah menjadi pengukuran sederhana pada citra biner (jumlah piksel foreground, komponen terhubung, bounding box). Tanpa itu, variasi pencahayaan, warna kertas, dan noise langsung bercampur ke ukuran yang kita pakai.

**Apa yang terjadi jika threshold terlalu tinggi atau terlalu rendah?** (lihat `threshold_too_low_high.png` dan `fg_ratio_vs_threshold.png`)
- **Terlalu rendah** (mis. T=80): hanya bagian tinta tergelap yang lolos. Goresan tipis/blur/pudar terputus atau hilang (`fg_ratio` 0.001 pada contoh), sehingga tanda tangan nyata dinilai ABSENT (**false negative**). Ini yang terjadi pada global T=127 di banyak varian.
- **Terlalu tinggi** (mis. T=252, di atas kecerahan kertas): kertas ikut menjadi foreground (`fg_ratio` = 1.0). Tanda tangan tenggelam, dan area kosong bisa dinilai PRESENT (**false positive**). Rentang tengah (T≈200–240) masih memberi bentuk yang wajar tetapi goresan makin tebal dan noise kertas mulai masuk.
- Otsu mencari T yang meminimalkan varians dalam-kelas, tetapi pada ROI yang tidak punya tinta (unimodal) ia tetap memotong noise kertas menjadi foreground. Itu sebabnya aturan memakai `ink_contrast` sebagai pengaman.

## Keterbatasan (penting dibaca)

- **Semua 9 citra berasal dari satu ijazah yang sama**, hanya dengan degradasi berbeda. Ini bukan data independen.
- **Tidak ada ijazah tanpa tanda tangan** pada data. Kelas "ABSENT" dibuat dari ROI kertas kosong dan teks tercetak pada ijazah yang sama. Untuk uji yang lebih kuat, tambahkan ijazah berbeda dan versi tanpa tanda tangan ke `data/`.
- **Ambang aturan disetel pada data uji yang sama** (tidak ada split train/test), jadi akurasi 100% untuk Otsu adalah *optimistis* dan bukan estimasi performa umum.
- ROI memakai koordinat tetap (relatif), sehingga mengasumsikan tata letak ijazah yang sama dan sudah tegak. Dokumen dengan tata letak lain butuh deteksi ROI otomatis.
- Pemisahan teks pendek (`teks_nomor`) dari tanda tangan bergantung pada `bbox_h`, heuristik yang sensitif terhadap ukuran ROI.

## Privasi

`data/` berisi ijazah asli (nama, tanggal lahir, NIM). Sebelum membuat repositori **publik**, hapus atau samarkan data tersebut, atau gunakan repositori privat.
