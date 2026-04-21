"""
ECG Preprocessing Pipeline for Chapman-Shaoxing Dataset
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, resample
# butter    → membuat filter Butterworth (desain koefisien filter)
# filtfilt  → menerapkan filter tanpa menggeser sinyal (forward-backward)
# resample  → mengubah panjang sinyal dengan metode Fourier

class ECGPreprocessor:
    """
    Preprocessing pipeline untuk ECG signals:
    1. Bandpass filtering (0.5-50 Hz)
    2. Normalization (Z-score)
    3. Resampling to fixed length
    """
    
    def __init__(self, 
                 target_length=5000,
                 sampling_rate=500,
                 lowcut=0.5, #batas bawah filter (buang noise <0.5 Hz)
                 highcut=50.0, #batas atas filter (buang noise >50 Hz)
                 filter_order=4, #makin tinggi makin tajam, tapi makin lambat
                 normalization='zscore'):
        """
        Args:
            target_length: Target signal length
            sampling_rate: Original sampling rate (Hz)
            lowcut: Low cutoff frequency for bandpass filter (Hz)
            highcut: High cutoff frequency for bandpass filter (Hz)
            filter_order: Butterworth filter order
            normalization: 'zscore', 'minmax', or 'none'
        """
        #simpan semua parameter ke objek agar bisa diakses fungsi lain
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.filter_order = filter_order
        self.normalization = normalization
        
    # Menyaring sinyal ECG, hanya frekuensi 0.5-50 Hz yang lewat
    # Tujuan: buang noise frekuensi rendah (gerak tubuh) dan tinggi (listrik/elektroda)    
    def bandpass_filter(self, ecg_signal):
        """
        Apply Butterworth bandpass filter
        """
        # Nyquist = setengah dari sampling rate, batas frekuensi tertinggi yang bisa direpresentasikan
        nyquist = 0.5 * self.sampling_rate
        
        # Normalisasi ke range 0-1 → 0.5 / 250 = 0.002
        # scipy butuh nilai ternormalisasi, bukan Hz langsung
        low = self.lowcut / nyquist
        # 50 / 250 = 0.2
        high = self.highcut / nyquist
        
        b, a = butter(self.filter_order, [low, high], btype='band')
        # butter() menghitung koefisien filter Butterworth
        # filter_order=4 → filter cukup tajam tanpa terlalu lambat
        # btype='band' → bandpass (loloskan frekuensi ANTARA low dan high)
        # b, a → koefisien numerator dan denominator filter

        filtered_signal = filtfilt(b, a, ecg_signal)
        # Terapkan filter ke sinyal ECG
        # filtfilt = filter 2 arah (maju + mundur)
        # Efek: tidak ada pergeseran fase, sinyal tidak terlambat/maju
        
        return filtered_signal
    
    
    # Menyeragamkan skala sinyal agar model tidak bias ke amplitude tertentu
    def normalize_signal(self, ecg_signal):
        """
        Normalize ECG signal
        """
        if self.normalization == 'zscore':
            # Z-score normalization (mean=0, std=1)
            mean = np.mean(ecg_signal) #rata-rata sinyal
            std = np.std(ecg_signal) #standar deviasi
            
            # Jika sinyal hampir flat (std ≈ 0), hindari pembagian nol
            # Cukup kurangi mean saja
            if std < 1e-6:
                return ecg_signal - mean
            
            normalized = (ecg_signal - mean) / std
            # Z-score: geser ke mean=0, skala ke std=1
            # Contoh: nilai 1.5mV dengan mean=0.5, std=0.5 → (1.5-0.5)/0.5 = 2.0

        #gadipake, pake nya zscore
        elif self.normalization == 'minmax':
            # Min-Max normalization (range: -1 to 1)
            min_val = np.min(ecg_signal)
            max_val = np.max(ecg_signal)
            
            # Avoid division by zero
            if (max_val - min_val) < 1e-6:
                return ecg_signal * 0
            
            normalized = 2 * (ecg_signal - min_val) / (max_val - min_val) - 1
            
        else:  # 'none'
            normalized = ecg_signal
            
        return normalized
    
    #tidak dipakai karena chapman = 5000, sudah sesuai target_length
    def resample_signal(self, ecg_signal):
        """
        Resample signal to fixed length
        """
        current_length = len(ecg_signal)
        
        if current_length == self.target_length:
            return ecg_signal
        
        elif current_length > self.target_length:
            # Downsample using Fourier method
            resampled = resample(ecg_signal, self.target_length)
            
        else:  # current_length < target_length
            # Zero padding
            pad_length = self.target_length - current_length
            resampled = np.pad(ecg_signal, (0, pad_length), mode='constant', constant_values=0)
        
        return resampled
    
    #fungsi utama yang dipanggil dataset_old (getitem)
    def process(self, ecg_signal):
        """
        Complete preprocessing pipeline
        
        Args:
            ecg_signal: Raw ECG signal (1D array)
            
        Returns:
            Preprocessed ECG signal
        """
        #menggabungkan semua langkn preprocessing secara berurutan
        # Step 1: Bandpass filtering : buang noise frekuensi tidak relevan
        filtered = self.bandpass_filter(ecg_signal)
        
        # Step 2: Normalization : seragamkan skala
        normalized = self.normalize_signal(filtered)
        
        # Step 3: Resampling to fixed length : samakan panjang
        resampled = self.resample_signal(normalized)
        
        return resampled


# Hanya jalan kalau file ini dijalankan langsung (bukan di-import)
if __name__ == "__main__":
    # Test preprocessing
    print("Testing ECG Preprocessor...")
    
    # Generate dummy signal
    fs = 500
    duration = 10
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate ECG-like signal
    ecg = np.sin(2 * np.pi * 1.2 * t)
    ecg += 0.3 * np.sin(2 * np.pi * 0.1 * t)
    ecg += 0.1 * np.random.randn(len(t))
    
    # Preprocess
    preprocessor = ECGPreprocessor()
    processed = preprocessor.process(ecg)
    
    print(f"Original length: {len(ecg)}")
    print(f"Processed length: {len(processed)}")
    print(f"Processed mean: {processed.mean():.6f}")
    print(f"Processed std: {processed.std():.6f}")

    print(ecg_signal[:10])       # 10 nilai pertama
    print(len(ecg_signal))       # panjang aslinya
    print(self.target_length)    # target panjang
    
    print("\n✓ Preprocessing test passed!")