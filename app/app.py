"""
ECG Arrhythmia Demo — Backend Python (Flask)
============================================
Jalankan: python app.py
Buka    : http://localhost:5000

Fungsi:
  - Serve file ecg_demo.html
  - Terima upload file .mat WFDB baru
  - Preprocessing: bandpass filter + z-score normalisasi
  - (Opsional) Inferensi model PyTorch jika checkpoint tersedia
"""

import os
import json
import numpy as np
import csv
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS  # <-- TAMBAHKAN INI
CORS(app) 
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt


app = Flask(__name__, static_folder='.')
 # <-- TAMBAHKAN INI (Agar GitHub Pages tidak diblokir browser)


# ─── Cek dependensi opsional ──────────────────────────────────────────────────
try:
    import scipy.io as sio
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠  scipy tidak ditemukan. Install: pip install scipy")

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("ℹ  PyTorch tidak ditemukan. Inferensi model dinonaktifkan.")

app = Flask(__name__, static_folder='.')

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
# Path ini mendeteksi folder 'app/' tempat app.py berada
APP_DIR = Path(__file__).resolve().parent

# Ganti CHECKPOINT_PATH agar naik satu tingkat (ke root project), baru masuk ke checkpoints
CHECKPOINT_PATH = APP_DIR / 'checkpoints' / 'best_model_ae_cn.pth'

# Tambahkan print ini untuk memastikan lokasinya di terminal saat Flask running
print(f"🔍 Flask mencari model di: {CHECKPOINT_PATH.resolve()}")
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
CLASS_NAMES = ['AFIB', 'GSVT', 'SB', 'SR']

SNOMED_MAP = {}
csv_path = APP_DIR.parent / 'data' / 'raw' / 'ConditionNames_SNOMED-CT.csv'
if csv_path.exists():
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get('Snomed_CT', '').strip()
            name = row.get('Full Name', '').strip()
            if code and name:
                SNOMED_MAP[code] = name
    print(f"✓ SNOMED mapping dimuat: {len(SNOMED_MAP)} kode")
else:
    print(f"⚠ CSV SNOMED tidak ditemukan di {csv_path}")

TARGET_LENGTH = 5000
FS = 500


# ─── GRU Model (inline — tidak perlu import dari models/) ─────────────────────
if TORCH_OK:
    class GRU_FeatureClassifier(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=1, num_classes=4):
            super().__init__()
            self.encoder = nn.GRU(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True
            )
            self.gn = nn.GroupNorm(1, hidden_size)
            self.bn = nn.BatchNorm1d(hidden_size)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x_t = x.transpose(1, 2)
            _, h_n = self.encoder(x_t)
            latent = h_n[-1]
            latent = self.gn(latent.unsqueeze(-1)).squeeze(-1)
            latent = self.bn(latent)
            return self.classifier(latent)


# ─── Load model jika ada ──────────────────────────────────────────────────────
model = None
if TORCH_OK and CHECKPOINT_PATH.exists():
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
        clf = GRU_FeatureClassifier(
            input_size=1, hidden_size=128, num_layers=1, num_classes=4
        )
        state = ckpt.get('clf_state_dict') or ckpt.get('model_state_dict')
        if state:
            clf.load_state_dict(state, strict=True)
            clf.eval()
            model = clf
            print(f"✓ Model dimuat dari {CHECKPOINT_PATH}")
        else:
            print("⚠  Checkpoint ditemukan tapi format tidak dikenali.")
    except Exception as e:
        print(f"⚠  Gagal memuat model: {e}")
else:
    if TORCH_OK:
        print(f"ℹ  Checkpoint tidak ditemukan di {CHECKPOINT_PATH}. Mode simulasi aktif.")


# ─── Preprocessing ────────────────────────────────────────────────────────────
def bandpass_filter(sig, fs=500, lo=0.5, hi=50, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, sig)


def zscore_normalize(sig):
    m, s = np.mean(sig), np.std(sig)
    return (sig - m) / (s + 1e-9)


def resample_signal(sig, target=5000):
    if len(sig) == target:
        return sig
    elif len(sig) > target:
        from scipy.signal import resample as sp_resample
        return sp_resample(sig, target)
    else:
        pad = target - len(sig)
        return np.pad(sig, (0, pad), mode='constant')


def preprocess_lead(raw_sig):
    """Full preprocessing pipeline: filter → normalize → resample."""
    filtered = bandpass_filter(raw_sig)
    normalized = zscore_normalize(filtered)
    resampled = resample_signal(normalized, TARGET_LENGTH)
    return resampled


def downsample_for_display(sig, n=500):
    """Downsample signal to n points for JSON transfer."""
    step = max(1, len(sig) // n)
    return [round(float(x), 4) for x in sig[::step][:n]]


def parse_mat(filepath):
    """
    Parse WFDB .mat file.
    Returns dict dengan raw, filtered, normalized per lead,
    plus metadata dari .hea jika ada.
    """
    mat = sio.loadmat(str(filepath))
    raw_data = mat['val'].astype(np.float32)   # shape (12, N) atau (N, 12)

    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    if raw_data.shape[0] > raw_data.shape[1]:
        raw_data = raw_data.T   # pastikan (leads, samples)

    n_leads, n_samples = raw_data.shape
    actual_leads = LEADS[:n_leads]

    # Coba baca .hea untuk gain & baseline
    hea_path = filepath.with_suffix('.hea')
    baselines = [0] * n_leads
    gains = [1000.0] * n_leads

    if hea_path.exists():
        with open(hea_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines[1:n_leads+1]):
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    gains[i] = float(parts[2].split('/')[0])
                    baselines[i] = int(parts[4])
                except Exception:
                    pass

    result = {'raw': {}, 'filtered': {}, 'normalized': {}}

    for i, lead in enumerate(actual_leads):
        sig_mv = (raw_data[i] - baselines[i]) / gains[i]
        filt = bandpass_filter(sig_mv)
        norm = zscore_normalize(filt)

        result['raw'][lead] = downsample_for_display(sig_mv)
        result['filtered'][lead] = downsample_for_display(filt)
        result['normalized'][lead] = downsample_for_display(norm)

    result['leads'] = actual_leads
    result['n_samples'] = n_samples
    result['n_leads'] = n_leads
    result['fs_display'] = 100
    result['original_fs'] = FS
    result['duration_sec'] = n_samples / FS

    return result


def run_inference(mat_filepath):
    """
    Jalankan model pada lead II (index 1).
    Return dict: predicted_class, probabilities, latent_preview
    """
    if not SCIPY_OK:
        return simulate_result()

    try:
        mat = sio.loadmat(str(mat_filepath))
        raw_data = mat['val'].astype(np.float32)
        if raw_data.ndim == 1:
            raw_data = raw_data.reshape(1, -1)
        if raw_data.shape[0] > raw_data.shape[1]:
            raw_data = raw_data.T

        # Ambil lead II (index 1) atau lead pertama
        lead_idx = 1 if raw_data.shape[0] > 1 else 0
        sig = raw_data[lead_idx]

        # Baca baseline dari .hea jika ada
        hea = mat_filepath.with_suffix('.hea')
        baseline, gain = 0, 1000.0
        if hea.exists():
            with open(hea) as f:
                lines = f.readlines()
            if len(lines) > lead_idx + 1:
                parts = lines[lead_idx + 1].strip().split()
                if len(parts) >= 5:
                    try:
                        gain = float(parts[2].split('/')[0])
                        baseline = int(parts[4])
                    except Exception:
                        pass

        sig_mv = (sig - baseline) / gain
        processed = preprocess_lead(sig_mv)

    except Exception as e:
        print(f"Error preprocessing: {e}")
        return simulate_result()

    if model is not None and TORCH_OK:
        try:
            # Ubah data numpy array menjadi tensor float
            tensor_data = torch.FloatTensor(processed) # shape: [5000]
            
            # Buat shape menjadi [Batch=1, Features=1, Seq_Len=5000] sesuai ekspektasi forward()
            x = tensor_data.unsqueeze(0).unsqueeze(0) 
            
            print(f"⚙️ [DEBUG] Shape tensor masuk ke model: {x.shape}") # Harus [1, 1, 5000]
            
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            print(f"📊 [DEBUG] Raw Probabilities: {probs}") # Melihat persentase mentah keempat kelas
            
            pred_idx = int(np.argmax(probs))
            return {
                'predicted': CLASS_NAMES[pred_idx],
                'probabilities': {c: float(round(probs[i], 4)) for i, c in enumerate(CLASS_NAMES)},
                'source': 'model'
            }
        except Exception as e:
            print(f"❌ Inference error: {e}")

    return simulate_result()


def simulate_result():
    """Simulasi hasil model (jika model tidak tersedia)."""
    probs_raw = np.array([0.72, 0.12, 0.09, 0.07]) + np.random.uniform(-0.02, 0.02, 4)
    probs_raw = np.clip(probs_raw, 0.01, None)
    probs_raw /= probs_raw.sum()
    return {
        'predicted': CLASS_NAMES[0],  # AFIB — sesuai ground truth JS00001
        'probabilities': {c: float(round(probs_raw[i], 4)) for i, c in enumerate(CLASS_NAMES)},
        'source': 'simulated'
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve halaman utama."""
    return send_file('ecg_demo.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Upload file .mat + .hea, return:
      - sinyal (raw/filtered/normalized per lead, 500 pts masing-masing)
      - hasil inferensi
      - metadata
    """
    if not SCIPY_OK:
        return jsonify({'error': 'scipy tidak terinstall. pip install scipy'}), 500

    mat_file = request.files.get('mat')
    hea_file = request.files.get('hea')

    if not mat_file:
        return jsonify({'error': 'File .mat diperlukan'}), 400

    # Simpan sementara
    tmp_dir = Path('/tmp/ecg_upload')
    tmp_dir.mkdir(exist_ok=True)
    stem = Path(mat_file.filename).stem

    mat_path = tmp_dir / f'{stem}.mat'
    hea_path = tmp_dir / f'{stem}.hea'

    mat_file.save(str(mat_path))
    if hea_file:
        hea_file.save(str(hea_path))

    # Parse sinyal
    try:
        signal_data = parse_mat(mat_path)
    except Exception as e:
        return jsonify({'error': f'Gagal parse .mat: {str(e)}'}), 400

    # Inferensi
    inference = run_inference(mat_path)

    # Metadata dari .hea jika ada
    meta = {'filename': stem, 'age': None, 'sex': None, 'dx_codes': [], 'dx_names': []}
    if hea_path.exists():
        with open(hea_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('#Age:'):
                    meta['age'] = line.split(':')[1].strip()
                elif line.startswith('#Sex:'):
                    meta['sex'] = line.split(':')[1].strip()
                elif line.startswith('#Dx:'):
                    codes = [c.strip() for c in line.split(':')[1].split(',')]
                    meta['dx_codes'] = codes
                    meta['dx_names'] = [SNOMED_MAP.get(c, f'Unknown ({c})') for c in codes]

    return jsonify({
        'signal': signal_data,
        'inference': inference,
        'meta': meta,
        'status': 'ok'
    })


@app.route('/api/status')
def status():
    return jsonify({
        'model_loaded': model is not None,
        'scipy': SCIPY_OK,
        'torch': TORCH_OK,
        'checkpoint': str(CHECKPOINT_PATH),
        'checkpoint_exists': CHECKPOINT_PATH.exists()
    })


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  ECG Arrhythmia Demo — Backend Python")
    print("=" * 55)
    print(f"  scipy  : {'✓' if SCIPY_OK else '✗ tidak ditemukan'}")
    print(f"  torch  : {'✓' if TORCH_OK else 'ℹ tidak ditemukan (simulasi aktif)'}")
    print(f"  model  : {'✓ dimuat' if model else 'simulasi (checkpoint tidak ada)'}")
    print("=" * 55)
    print("  Buka  : http://localhost:5000")
    print("  Henti : Ctrl+C")
    print("=" * 55)

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)