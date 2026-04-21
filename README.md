# Klasifikasi Arritmia Menggunakan GRU-CN Autoencoder

Repository ini berisi penelitian  mengenai klasifikasi sinyal fisiologis (ECG) menggunakan arsitektur **Gated Recurrent Unit (GRU)** dengan **Continual Normalization (CN)** dan pendekatan **Autoencoder**.

## 📊 Hasil Penelitian
Model ini telah diuji dan mencapai performa berikut:
* **Accuracy:** 92.57%
* **Macro F1-Score:** 0.9181
* **AUC Macro:** 99.00%

## 🚀 Fitur Utama
* **GRU-CN Architecture:** Mengatasi masalah internal covariate shift pada data sekuensial.
* **Two-Stage Learning:** Tahap pra-pelatihan menggunakan Autoencoder (Unsupervised) diikuti dengan klasifikasi.
* **Preprocessing:** Normalisasi dan pemrosesan data numerik dari dataset Chapman-Shaoxing.

## 📂 Struktur Folder
* `models/`: Definisi arsitektur GRU-CN dan Autoencoder.
* `scripts/`: Script untuk pelatihan model (Baseline, BN, GN, dan CN).
* `utils/`: Fungsi pembantu untuk preprocessing sinyal dan metrik evaluasi.
* `data/processed/`: Metadata dan split dataset yang telah diproses.

## 🛠️ Cara Penggunaan
1. Install dependensi (Python 3.x).
2. Jalankan pelatihan dengan:
   ```bash
   python scripts/train_gru_ae_cn_gn.py
