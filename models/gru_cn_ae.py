"""
GRU Autoencoder + Classifier

Includes:
- `GRU_Autoencoder`: encoder+decoder for unsupervised pretraining
- `GRU_FeatureClassifier`: feature extractor (encoder) + classifier that applies
  BatchNorm followed by GroupNorm (continual normalization)

This file follows the style of existing `gru_cn.py` for compatibility.
"""

import torch
import torch.nn as nn


class GRU_Autoencoder(nn.Module): #turunan dari nn.Module
    """Simple GRU-based sequence autoencoder.

    Encoder compresses sequence into a latent vector (last hidden state).
    Decoder tries to reconstruct the input sequence from the latent vector.
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=1,
                 bidirectional=False): #Parameter default: 1 channel input, 128 hidden unit, 1 layer, tidak bidirectional.
        super(GRU_Autoencoder, self).__init__()#Memanggil konstruktor dari kelas induk nn.Module untuk inisialisasi dasar model.

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1 #klo bidirectional, arah ada 2 (forward + backward)

        #GRU encoder — baca urutan input dan hasilkan hidden state sebagai "ringkasan".
        self.encoder = nn.GRU(
            input_size=input_size,      #1 channel
            hidden_size=hidden_size,    #128
            num_layers=num_layers,      #1 layer
            batch_first=True,
            bidirectional=bidirectional,    #False
        )

        ##Layer linear untuk memproyeksikan latent vector sebelum masuk decoder.
        self.latent_proj = nn.Linear(hidden_size * self.num_directions,
                                     hidden_size * self.num_directions)

        #GRU decoder — tugasnya merekonstruksi ulang urutan asli dari latent vector.
        self.decoder = nn.GRU(
            input_size=hidden_size * self.num_directions,
            hidden_size=hidden_size,    #128
            num_layers=1,               #1 layer
            batch_first=True,
            bidirectional=False,
        )

        #Proyeksikan output decoder kembali ke dimensi input asli (misal 1 channel).
        self.output_proj = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        """Encode and decode.

        Args:
            x: tensor [batch, channels(=1), seq_len]

        Returns:
            recon: reconstructed x with same shape
            latent: latent vector [batch, latent_dim]
        """
        #Ubah shape dari [batch, channel, seq_len] → [batch, seq_len, channel] karena GRU maunya begitu.
        x_t = x.transpose(1, 2)

        # Jalankan encoder, h_n = hidden state terakhir (isinya "ringkasan" sequence).
        enc_out, h_n = self.encoder(x_t)

        # Take last timestep hidden representation
        if self.bidirectional:
            # h_n: (num_layers*2, batch, hidden)
            # concat forward and backward for top layer
            # take last layer forward and backward
            last_f = h_n[-2]
            last_b = h_n[-1]
            latent = torch.cat([last_f, last_b], dim=1)
        else:
            latent = h_n[-1] #Ambil hidden state dari layer terakhir sebagai latent vector.

        # Proyeksikan latent, lalu squeeze lewat tanh supaya nilainya antara -1 dan 1.
        dec_init = torch.tanh(self.latent_proj(latent))

        # Repeat latent as input for every timestep
        seq_len = x_t.size(1)
        #Latent vector di-repeat sebanyak seq_len timestep — jadi input decoder di tiap langkah sama.
        dec_in = dec_init.unsqueeze(1).expand(-1, seq_len, -1)

        #Jalankan decoder → proyeksi ke dimensi asli → balik shape ke [batch, channel, seq_len].
        dec_out, _ = self.decoder(dec_in)
        recon = self.output_proj(dec_out)
        recon = recon.transpose(1, 2)

        return recon, latent #Kembalikan hasil rekonstruksi dan latent vector-nya.


class GRU_FeatureClassifier(nn.Module):
    """Feature extractor + classifier.

    Feature extractor is the encoder part (re-uses GRU encoder design).
    Classifier applies combined BatchNorm + GroupNorm on features before FC.
    """

    #inisiasliasi layer dan parameter model, dengan kondisi bobot yang diisi oleh pytorch random default, tapi kita akan override dengan _init_weights() nanti.
    def __init__(self, input_size=1, hidden_size=128, num_layers=1,
                 num_classes=10, bidirectional=False, groups=8, dropout=0.3):
        super(GRU_FeatureClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.latent_dim = hidden_size * self.num_directions

        # Encoder GRU (same as AE encoder)
        self.encoder = nn.GRU(
            input_size=input_size,           # 1 channel/1 lead
            hidden_size=hidden_size,         # 128
            num_layers=num_layers,           # 1 layer
            batch_first=True,
            bidirectional=bidirectional,     # False
        )

        # Continual normalization layers applied to latent vector
        self.gn = nn.GroupNorm(num_groups=min(groups, self.latent_dim), 
                               num_channels=self.latent_dim) 
        self.bn = nn.BatchNorm1d(self.latent_dim)                           #!28 features
       

        self.dropout = nn.Dropout(dropout) #dropout untuk cegah overfitting                                  #0.3 dropout    
        self.classifier = nn.Linear(self.latent_dim, num_classes)#Layer terakhir: mapping dari fitur (128) ke jumlah kelas (4). = Fully connected (etiap 128 neuron terhubung ke semua 4 output)

        self._init_weights()

    #Inisialisasi bobot dengan teknik yang tepat agar training lebih stabil.
    #Parameter (yg di olah) = semua angka/bobot yang dipelajari model saat training []
    #analogi -> Model = Murid baru masuk sekolah ; Parameter = pengetahuan awal si murid
    #Semua ini cuma cara mengisi angka awal ke dalam bobot. Bobot itu pada dasarnya cuma tabel angka (matriks).
    def _init_weights(self): 
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data) #utk weight input
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)#utk weight recurrent
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
            elif 'classifier.weight' in name or 'output_proj.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'classifier.bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x): #Sama seperti encoder AE — ambil latent dari hidden state terakhir.
        """Forward classifier.

        Args:
            x: [batch, channels=1, seq_len]

        Returns:
            logits: [batch, num_classes]
        """
        x_t = x.transpose(1, 2)

        enc_out, h_n = self.encoder(x_t)

        if self.bidirectional:
            last_f = h_n[-2]
            last_b = h_n[-1]
            latent = torch.cat([last_f, last_b], dim=1)
        else:
            latent = h_n[-1] #feature extractor-nya sama dengan encoder AE, jadi kita ambil latent vector dari hidden state terakhir.

        # BN requires 2D input: [batch, features]
        # Fix:
        latent_gn = self.gn(latent.unsqueeze(-1)).squeeze(-1)   # GN dulu #Tambah dimensi dummy dulu (unsqueeze) supaya GroupNorm bisa jalan, lalu hapus lagi (squeeze).
        latent_bn = self.bn(latent_gn)                           # BN kedua, input dari latent_GN

        #Dropout → classifier → kembalikan logits (skor mentah sebelum softmax).
        h = self.dropout(latent_bn)
        logits = self.classifier(h)
        return logits


if __name__ == "__main__":
    # Smoke test
    batch_size = 4
    seq_len = 500
    input_size = 1
    num_classes = 5

    ae = GRU_Autoencoder(input_size=input_size, hidden_size=64, num_layers=1)
    clf = GRU_FeatureClassifier(input_size=input_size, hidden_size=64, num_layers=1,
                                num_classes=num_classes, bidirectional=False)

    x = torch.randn(batch_size, input_size, seq_len) #input dummy
    recon, latent = ae(x) #Jalankan autoencoder untuk dapatkan rekonstruksi dan latent vector
    logits = clf(x) #Jalankan classifier untuk dapatkan logits (skor kelas) dari input yang sama

    print('x', x.shape)
    print('recon', recon.shape)
    print('latent', latent.shape)
    print('logits', logits.shape)


"""
# Jawaban Satu Per Satu

---

## 1. `forward` itu apa?

`forward` = **resep cara data mengalir** di dalam model.

Di PyTorch, setiap model punya `forward`. Saat kamu tulis `model(x)`, PyTorch otomatis manggil `forward(x)`.

**Bedanya AE vs Classifier:**
```
Autoencoder forward:        Classifier forward:
Input → Encode → Decode     Input → Encode → Normalize → Classify
Tujuan: rekonstruksi ulang  Tujuan: prediksi kelas
Output: recon + latent      Output: logits (skor kelas)
```

---

## 2. `self.` itu apa?

`self` = **"milik si object ini"**

Analoginya kayak nama orang:
```python
self.encoder = nn.GRU(...)   # encoder MILIK model ini
self.dropout = nn.Dropout()  # dropout MILIK model ini
```
Kalau ga pakai `self`, variabelnya cuma hidup di dalam fungsi itu doang dan hilang setelahnya. Pakai `self` berarti bisa diakses dari mana saja di dalam class.

---

## 3. `_init_weights` buat apa?

Fungsinya: **set nilai awal bobot** supaya training lebih stabil.

```python
xavier_uniform_   → bobot input GRU, mencegah nilai terlalu besar/kecil
orthogonal_       → bobot recurrent, mencegah gradient hilang/meledak
constant_(0)      → bias diset nol, titik awal netral
```

**Kalau ga dipake?** PyTorch tetap inisialisasi otomatis (random), tapi:
- Training bisa lebih lambat
- Bisa lebih mudah *vanishing/exploding gradient*
- Hasilnya tidak deterministik dan bisa kurang stabil

Intinya: **bisa jalan, tapi lebih berisiko**.

---

## 4. Softmax-nya dimana?

**Tidak ada di sini**, dan itu **disengaja!** ✅

```python
logits = self.classifier(h)  # output mentah, belum di-softmax
return logits
```

Karena PyTorch punya `nn.CrossEntropyLoss` yang sudah **include softmax di dalamnya**. Kalau softmax ditambah lagi di model, jadinya double → salah.

```
Model output: logits → CrossEntropyLoss (softmax di dalam) → loss
                                    atau
Model output: logits → softmax manual → NLLLoss
```

---

## 5. Smoke Test itu apa?

**Tes cepat "apakah tidak meledak?"** — bukan tes akurasi.

```python
x = torch.randn(4, 1, 500)   # data dummy asal-asalan
recon, latent = ae(x)         # coba jalankan
logits = clf(x)               # coba jalankan
print(recon.shape)            # cek shape bener ga
```

Tujuannya cuma mastiin:
- Tidak ada error
- Shape input/output sesuai
- Semua layer nyambung

Kayak test drive mobil di parkiran sebelum bawa ke jalan tol.

---

## 6. `nn.Module`, `nn.GRU` itu apa?

`nn` = `torch.nn` = **kotak perkakas neural network bawaan PyTorch**

| | Fungsi |
|---|---|
| `nn.Module` | "cetakan dasar" semua model PyTorch. Wajib di-inherit |
| `nn.GRU` | Layer GRU siap pakai, tinggal tunjuk ukurannya |
| `nn.Linear` | Layer fully connected (perkalian matriks) |
| `nn.BatchNorm1d` | Normalisasi per batch |
| `nn.Dropout` | Matikan neuron secara random saat training |

Analogi: `nn` itu seperti LEGO set. `nn.Module` adalah papan dasarnya, yang lain adalah potongan LEGO yang kamu tempel.

---

## 7. Penjelasan blok bidirectional yang membingungkan

### Konteks dulu: GRU bidirectional itu apa?

```
Normal GRU:      A → B → C → D   (baca kiri ke kanan)
Bidirectional:   A → B → C → D   (forward)
                 A ← B ← C ← D   (backward)
```
Dengan bidirectional, model baca dari **dua arah**, lalu digabung.

### Sekarang kodenya:

```python
if self.bidirectional:
    # h_n bentuknya: [num_layers * 2, batch, hidden]
    # *2 karena ada 2 arah
    # misal 1 layer bidirectional → h_n shape: [2, batch, 128]
    
    last_f = h_n[-2]  # index -2 = forward  direction layer terakhir
    last_b = h_n[-1]  # index -1 = backward direction layer terakhir
    
    latent = torch.cat([last_f, last_b], dim=1)
    # gabungkan → [batch, 256]  (128 + 128)
else:
    latent = h_n[-1]
    # cuma 1 arah → ambil layer terakhir saja → [batch, 128]
```

### Lanjutannya:

```python
dec_init = torch.tanh(self.latent_proj(latent))
# latent → Linear → tanh
# tanh memaksa nilai jadi -1 sampai 1
# supaya decoder punya "starting point" yang terkontrol
```

```python
seq_len = x_t.size(1)         # ambil panjang sequence, misal 5000

dec_in = dec_init             # shape: [batch, 128]
       .unsqueeze(1)          # tambah dimensi → [batch, 1, 128]
       .expand(-1, seq_len, -1) # duplikasi → [batch, 500, 128]
```

**Kenapa di-repeat?** Karena decoder butuh input di **setiap timestep** (500 langkah), tapi kita cuma punya 1 latent vector. Jadi latent vector yang sama dikasih ke decoder di setiap langkah.

```
Latent [batch, 128]
    ↓ unsqueeze
[batch, 1, 128]
    ↓ expand ke 5000 timestep
[batch, 5000, 128]  ← ini yang masuk decoder
```
"""