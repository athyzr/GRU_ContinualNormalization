# ECG Arrhythmia Demo
### GRU Autoencoder + Continual Normalization (BN+GN)

Demo interaktif berbasis web untuk memvisualisasikan pipeline ECG arrhythmia classification penelitianmu.

---

## 📁 Struktur File

```
ecg_demo.html        ← Aplikasi utama (bisa dibuka langsung di browser)
app.py               ← Backend Python/Flask (untuk upload file .mat baru)
README.md            ← File ini
```

---

## 🚀 Cara Menjalankan

### Mode 1 — Tanpa Backend (paling simpel)
Cukup buka `ecg_demo.html` langsung di browser:
```
Klik 2× pada ecg_demo.html
```
Data ECG pasien **JS00001** sudah ter-embed. Kamu bisa:
- Lihat sinyal semua 12 lead
- Switch antara Raw / Filtered / Normalized
- Jalankan animasi pipeline preprocessing
- Lihat hasil prediksi (simulasi)

---

### Mode 2 — Dengan Backend Python (untuk upload file .mat baru)

**Install dependensi:**
```bash
pip install flask scipy numpy
pip install torch  # opsional, untuk inferensi model nyata
```

**Jalankan backend:**
```bash
python app.py
```

**Buka browser:**
```
http://localhost:5000
```

**Upload file .mat:**
- Klik zona upload di sidebar kiri
- Upload file `.mat` (dan `.hea` untuk metadata lengkap)
- Backend akan preprocessing & kirim data ke frontend

---

## 🔬 Pipeline yang Didemonstrasikan

```
File .mat (WFDB)
    │
    ├─ [Step 1] Load Sinyal ECG
    │       wfdb.rdrecord() → 12-lead × 5000 pts (500 Hz, 10 detik)
    │
    ├─ [Step 2] Bandpass Filter
    │       Butterworth order-4, 0.5–50 Hz
    │       Hilangkan noise gerak tubuh & interferensi listrik
    │
    ├─ [Step 3] Z-Score Normalisasi
    │       (x - mean) / std per lead
    │       Seragamkan skala antar lead & pasien
    │
    ├─ [Step 4] GRU Encoder
    │       [batch, 1, 5000] → latent vector [batch, 128]
    │       Hidden state terakhir sebagai representasi temporal
    │
    ├─ [Step 5] Continual Normalization
    │       GroupNorm → BatchNorm
    │       Stabilisasi distribusi fitur
    │
    └─ [Step 6] Klasifikasi
            Linear(128 → 4) → Softmax
            4 Kelas: AFIB | GSVT | SB | SR
```

---

## 🏷️ Kelas Aritmia

| Kode | Nama Lengkap | Keterangan |
|------|-------------|------------|
| **AFIB** | Atrial Fibrillation | Fibrilasi atrium — irama tidak teratur |
| **GSVT** | Generalized SVT | Takikardia supraventrikular |
| **SB** | Sinus Bradycardia | Detak jantung lambat < 60 bpm |
| **SR** | Sinus Rhythm | Irama sinus normal |

---

## 📊 Data Pasien Demo (JS00001)

| Field | Nilai |
|-------|-------|
| Usia | 85 tahun |
| Jenis Kelamin | Laki-laki |
| Lead | 12 lead (I, II, III, aVR, aVL, aVF, V1–V6) |
| Sampling Rate | 500 Hz |
| Durasi | 10 detik (5000 sample) |
| Diagnosis Klinis | 164889003 — Atrial Fibrillation |
| | 59118001 — Right Bundle Branch Block |
| | 164934002 — T-wave Abnormality |

Ground truth: **AFIB** → Model memprediksi **AFIB** ✓

---

## 🔌 API Backend (saat menjalankan app.py)

**POST `/api/analyze`**
```
Form data:
  mat  : file .mat (wajib)
  hea  : file .hea (opsional, untuk metadata)

Response JSON:
  signal.raw          : sinyal asli per lead (500 pts)
  signal.filtered     : setelah bandpass filter
  signal.normalized   : setelah z-score
  inference.predicted : kelas prediksi (AFIB/GSVT/SB/SR)
  inference.probs     : probabilitas tiap kelas
  meta.age, meta.sex  : metadata pasien
```

**GET `/api/status`**
```
Cek apakah model tersedia & dependensi lengkap
```

---

## 🧠 Menggunakan Model Terlatih

Jika kamu sudah melatih model dengan `train_gru_ae_cn_gn.py`, salin checkpoint:
```bash
cp checkpoints/best_model_ae_cn.pth ./checkpoints/best_model_ae_cn.pth
```
Lalu jalankan `python app.py` — backend akan otomatis memuat model dan menjalankan inferensi nyata (bukan simulasi).

---

## ⚠️ Catatan

- Mode tanpa backend menggunakan **simulasi** hasil model (probabilitas pre-set)
- Untuk inferensi nyata, perlu model terlatih + PyTorch + Flask backend
- File `.hea` sangat dianjurkan disertakan bersama `.mat` agar konversi ADC → mV akurat