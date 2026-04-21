"""
IMPROVED GRU Autoencoder + Continual Normalization Training
with proper scheduler, early stopping, and gradient clipping.

metrics.py ; dataset_old.py ; gru_cn_ae.py ; preprocessing.py 
"""

import sys
from pathlib import Path
# Menambahkan folder induk ke sistem agar Python bisa menemukan modul ku sendiri (seperti models atau utils) meskipun berada di folder berbeda.
sys.path.append(str(Path(__file__).resolve().parents[1])) 

import argparse #membuat input perintah di terminal (seperti menentukan jumlah epochs atau learning rate
import time
import os
import json
import datetime
import shutil
 
import torch # Inti dari framework PyTorch untuk membangun saraf tiruan.
import numpy as np #Operasi numerik
import torch.nn as nn #Inti dari framework PyTorch untuk membangun saraf tiruan.
import torch.optim as optim #optimizier (adam)
from tqdm import tqdm #Library untuk memunculkan progress bar yang cantik saat loading/training.

import utils.dataset_old as dataset_old
from models.gru_cn_ae import GRU_Autoencoder, GRU_FeatureClassifier
from utils.metrics import calculate_metrics, get_classification_report
from utils.preprocessing import ECGPreprocessor
from utils.visualization import (
    plot_learning_curves, 
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_curves
)

# Phase 1 : Belajar untuk bentuk sinyal
def train_ae(ae, loader, device, epochs=10, lr=1e-3):
    ae.to(device) #memindahkan model ke device yang ditentukan (CPU atau GPU)
    ae.train() # Mengatur model ke mode pelatihan (mengaktifkan Dropout, BatchNorm, dll)
    opt = optim.Adam(ae.parameters(), lr=lr) #Menggunakan optimizer Adam untuk memperbarui bobot model 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=2, factor=0.5
    ) #Scheduler untuk menurunkan learning rate jika loss tidak membaik. jika loss turun selama 2 eopch LR dikurangi 0.5
    
    #Menghitung seberapa mirip sinyal asli dengan sinyal hasil rekonstruksi (Mean Squared Error).
    criterion = nn.MSELoss() 

    best_loss = float('inf') #menyimpan loss terbaik
    patience_counter = 0
    patience = 3 #Jika model tidak membaik selama 3 epoch → stop.

    #Loop epoch
    for epoch in range(epochs):
        t0 = time.time() #mencatat waktu mulai epoch untuk menghitung durasi
        running = 0.0 #menyiman total loss selama epoch untuk menghitung rata-rata di akhir
        pbar = tqdm(loader, desc=f'AE Epoch {epoch+1}/{epochs}') #progress bar untuk iterasi data
        
        #Loop Batch training autoencoder
        for x, _ in pbar: #x= input ECG, _=label (tidak digunakan untuk AE)
            x = x.to(device) #memindahkan data batch ke device yang sama dengan model (CPU/GPU)
            recon, _ = ae(x) #forward pass melalui autoencoder, menghasilkan rekonstruksi dan latent vector
            loss = criterion(recon, x) #menghitung loss antara rekonstruksi dan input asli
            opt.zero_grad() #menghapus gradien sebelumnya
            loss.backward() #backpropagation untuk menghitung gradien berdasarkan loss

            #Gradien : Dia mengukur seberapa besar perubahan Error (Loss) jika mengubah sedikit saja Bobot (Weight) model.
            # Gradient clipping, mencegah gradien terlalu besar yang bisa menyebabkan pelatihan tidak stabil.
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            
            opt.step() #update weights model berdasarkan gradien yang dihitung
            running += loss.item() * x.size(0) #menghitung total loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'}) #menampilkan loss saat ini di progress bar
        
        #Hitung loss epoch
        avg_loss = running / len(loader.dataset) #rata-rata loss
        elapsed = time.time() - t0 #durasi epoch
        print(f"AE Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Time: {elapsed:.1f}s")
        
        # Scheduler step, updaate learning rate berdasarkan rata-rata loss. Jika loss tidak membaik, scheduler akan menurunkan LR.
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Early stopping check
        #jika rata-rata membaik, reset patience
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            #jika tidak membaik stop traning
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping AE at epoch {epoch+1}")
                break


# Fungsi ini digunakan untuk melatih model klasifikasi aritmia ECG setelah encoder dipelajari oleh autoencoder.
def train_classifier(clf, train_loader, val_loader, device, epochs=30, lr=5e-4,
                     checkpoint_dir=None):

    clf.to(device) #clf = model classifier, memindahkan model ke device yang ditentukan (CPU atau GPU)
    opt = optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-5) #optimizer adam dengna weight decay (untuk mengurangi overfitting) dengan learning rate yang lebih rendah untuk stabilitas pelatihan classifier. LR yang terlalu tinggi bisa menyebabkan pelatihan tidak stabil, terutama setelah transfer learning dari autoencoder.
    
    # menurunkan learning rate jika validation accuracy tidak meningkat
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=3, factor=0.5
    )
    
    #Loss function untuk klasifikasi multi-class.
    criterion = nn.CrossEntropyLoss()
    
    #menyimpan performa terbaik untuk validasi
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5  
    
    #menyimpan history training 
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    #Loop Epoch 
    for epoch in range(epochs):
        # ===== TRAINING PHASE =====
        clf.train() #model masuk mode training

        #variabel untuk mengitung performa training
        running_loss = 0.0
        correct = 0
        total = 0
        
        #progress bar training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        #loop batch training 
        #mengambil batch data (x=ECG input, y=label kelas)
        for x, y in pbar:
            # memindahkan data ke gpu/cpu yang sama dengan model
            x = x.to(device)
            y = y.to(device).view(-1) #view(-1) digunakan untuk memastikan bentuk label benar.
            
            logits = clf(x) #forward pass melalui classifier untuk mendapatkan output logit (sebelum softmax)
            # forward pass = proses data masuk ke modeluntuk menghasilkan prediksi
            loss = criterion(logits, y) #menghitung loss 
            
            #backpropagation dan update weights
            #backpro = belajar dri kesalahan, error dikirim dri output ke input
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping untuk stabilitas training
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=5.0)
            
            #update weight model 
            opt.step()
            
            running_loss += loss.item() * x.size(0) #menjumlahkan loss batch
            _, preds = torch.max(logits, 1) #mengambil prediksi kelsas dengan nilai logit tertinggi
            correct += (preds == y).sum().item() #mgnhitung jumlah prediksi benar
            total += y.size(0) #menghitung total sampel yang diproses
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        #hitung hasil training rata-rata loss dan akurasi untuk epoch ini
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ===== VALIDATION PHASE =====
        #masuk mode evaluasi (nonaktifkan dropout)
        clf.eval()
        #menyimpan hasil validsi
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad(): #Tidak menghitung gradient (lebih cepat dan hemat memori).
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            
            # loop data validasi 
            for x, y in pbar_val:
                #pindah data ke device yang sama dengan model
                x = x.to(device)
                y = y.to(device).view(-1)
                
                #forward pass untuk mendapatkan logit dan menghitung loss validasi
                logits = clf(x)
                loss = criterion(logits, y)
                
                val_running_loss += loss.item() * x.size(0) #akumulasi loss 
                preds = torch.argmax(logits, dim=1) #prediksi kelas dengan nilai logit tertinggi
                #menyimpan hasil 
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
        
        #hitung metrics validasi
        val_loss = val_running_loss / len(val_loader.dataset)
        # menggabungkan semua prediksi dan label di semua batch untuk menghitung metrik keseluruhan pada set validasi
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        #menghtiung metrik
        metrics = calculate_metrics(y_true.numpy(), y_pred.numpy(), num_classes=None)
        
        val_acc = metrics['accuracy']
        val_f1 = metrics.get('f1_macro', metrics.get('f1', 0.0))
        
        # simpan history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"  LR: {opt.param_groups[0]['lr']:.6f}")
        
        #scheduler update, Update learning rate berdasarkan validation accuracy.
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # jika model membaik, simpan checkpoint dan reset patience
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  ✓ New best! Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}")
            
            # Save best model (to experiment folder if provided)
            save_path = 'checkpoints/best_model_ae_cn.pth'
            if checkpoint_dir is not None:
                save_path = str(Path(checkpoint_dir) / 'best_model_ae_cn.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': clf.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'history': history
            }, save_path)
        else:
            #jika tidak membaik, tingkatkan counter untuk early stopping
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
        
        print("-" * 70)
    
    return best_val_acc, best_val_f1, history


#mengatur seluruh pipeline training autoencoder + classifier, termasuk setup eksperimen, logging, dan evaluasi akhir.
def main():
    parser = argparse.ArgumentParser() #untuk mengatur argumen yang bisa diberikan saat menjalankan script dari terminal, seperti menentukan jumlah epoch, learning rate, batch size, dll.
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ae_epochs', type=int, default=10,
                        help='Autoencoder pretraining epochs')
    parser.add_argument('--clf_epochs', type=int, default=30,
                        help='Classifier training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--selected_lead', type=str, default='1',
                        help="Lead to use for training; set 'all' for 12‑lead input")
    parser.add_argument('--ae_lr', type=float, default=1e-3,
                        help='Autoencoder learning rate')
    parser.add_argument('--clf_lr', type=float, default=5e-4,
                        help='Classifier learning rate (lower = more stable)')
    args = parser.parse_args() # membca parameter dri terminal

    #mengaktifkan gpu
    device = torch.device(args.device)
    
    # project paths
    project_root = Path(__file__).parent.parent
    data_dir = str(project_root / 'data' / 'raw' / 'WFDBRecords')
    metadata_path = str(project_root / 'data' / 'processed' / 'metadata_full.csv')
    target_classes_path = str(project_root / 'data' / 'processed' / 'target_classes.csv')
    condition_names_path = str(project_root / 'data' / 'raw' / 'ConditionNames_SNOMED-CT.csv')

    # experiment directory (timestamped)
    exp_root = project_root / 'experiments'
    exp_root.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'gru_ae_cn_{timestamp}'
    exp_dir = exp_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)

    # also keep a copy in the global checkpoints folder for backward compatibility
    default_ckpt_dir = project_root / 'checkpoints'
    default_ckpt_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("IMPROVED GRU AUTOENCODER + COMBINED NORM TRAINING")
    print("="*70)
    print(f"Device         : {device}")
    print(f"Batch size     : {args.batch_size}")
    print(f"AE epochs      : {args.ae_epochs}")
    print(f"CLF epochs     : {args.clf_epochs}")
    print(f"AE LR          : {args.ae_lr}")
    print(f"CLF LR         : {args.clf_lr} (LOWER for stability)")
    print("="*70 + "\n")

    # Create preprocessor
    preprocessor = ECGPreprocessor()

    # Load dataset
    print("Loading data...")
    train_loader, val_loader, test_loader, info = dataset_old.create_dataloaders( #membuat dataloader dri modul dataset_old, yang akan digunakan untuk melatih model. 
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        superclass_mode=True,
        condition_names_path=condition_names_path,
        use_class_sampler=True, #menggunakan class sampler untuk menangani ketidakseimbangan kelas (weightedRandomSampler)
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        selected_lead=args.selected_lead
    )

    print(f"\n✓ Dataset loaded:")
    print(f"  Classes: {info['class_names']}")
    print(f"  Train: {info['train_size']:,} samples")
    print(f"  Val: {info['val_size']:,} samples\n")

    #membuat model autoencoder dan classifier. Autoencoder akan belajar merekonstruksi sinyal ECG, sedangkan classifier akan belajar mengklasifikasikan kondisi aritmia berdasarkan fitur yang dipelajari oleh encoder autoencoder.
    ae = GRU_Autoencoder(
        input_size=info.get('num_leads', 1), #Bukti hanya pakai 1 lead
        hidden_size=128,
        num_layers=1
    )
    
    clf = GRU_FeatureClassifier(
        input_size=info.get('num_leads', 1),
        hidden_size=128,
        num_layers=1,
        num_classes=info['num_classes']
    )

    # ===== PHASE 1: AUTOENCODER PRETRAINING =====
    print("\n" + "="*70)
    print("PHASE 1: AUTOENCODER PRETRAINING (with scheduler)")
    print("="*70)
    #training autoencoder
    train_ae(ae, train_loader, device, epochs=args.ae_epochs, lr=args.ae_lr)

    # ===== PHASE 2: TRANSFER WEIGHTS =====
    print("\n" + "="*70)
    print("PHASE 2: TRANSFERRING ENCODER WEIGHTS")
    print("="*70)
    ae_state = ae.encoder.state_dict() #mengambil bobot encoder dari autoencoder yang sudah dilatih
    clf_state = clf.encoder.state_dict() #transfer learning dari encoder autoencoder ke encoder classifier. Hanya bobot yang memiliki bentuk yang sama yang akan ditransfer, sehingga memastikan kompatibilitas antara kedua model.
    transferred = 0
    for k in clf_state.keys():
        if k in ae_state and ae_state[k].shape == clf_state[k].shape:
            clf_state[k] = ae_state[k].clone()
            transferred += 1
    clf.encoder.load_state_dict(clf_state)
    print(f"✓ Transferred {transferred} weight tensors\n")

    # ===== PHASE 3: CLASSIFIER TRAINING =====
    print("="*70)
    print("PHASE 3: CLASSIFIER TRAINING (with scheduler & early stopping)")
    print("="*70)
    #trrainign classifier dengan scheduler dan early stopping
    best_acc, best_f1, history = train_classifier(
        clf, train_loader, val_loader, device,
        epochs=args.clf_epochs, lr=args.clf_lr,
        checkpoint_dir=exp_dir / 'checkpoints'
    )

    # ===== SAVE FINAL MODEL =====
    # write final checkpoint in both experiment and default folders
    exp_ckpt_dir = exp_dir / 'checkpoints'
    exp_ckpt_dir.mkdir(exist_ok=True)

    final_exp_path = exp_ckpt_dir / 'gru_ae_cn_improved.pth'
    #menyimpan model akhir, informasi eksperimen, dan history training ke dalam file checkpoint. 
    torch.save({
        'ae_state_dict': ae.state_dict(),
        'clf_state_dict': clf.state_dict(),
        'info': info,
        'best_val_acc': best_acc,
        'best_val_f1': best_f1,
        'history': history,
        'args': vars(args)
    }, final_exp_path)
    # also duplicate to top‑level checkpoints for compatibility
    shutil.copy(final_exp_path, default_ckpt_dir / final_exp_path.name)

    #evaluasi akhir model pada set validasi untuk mendapatkan metrik lengkap,
    clf.eval()
    #membuat list kosong untuk menyimpan prediksi, label dan probabilitias tiap kelas
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad(): #tidak menghitung gradient
        for x, y in val_loader: #loop data validasi
            #pindah ke gpu
            x = x.to(device) 
            y = y.to(device).view(-1)
            #forward pass untuk mendapatkan logit, prediksi kelas, dan probabilitas kelas (softmax) untuk setiap sampel di set validasi.
            logits = clf(x)
            preds = torch.argmax(logits, dim=1) #mengambil kelas dengan nilai tertinggi
            probs = torch.softmax(logits, dim=1) #mnghitung logits menjadi probabilitas kelas dengan softmax
            all_preds.append(preds.cpu()) #menyimpan prediksi kelas ke list (dipindahkan ke CPU untuk menghemat memori GPU)
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
    #menggabungkan semua batch prediksi menjadi satu arrays besar untuk menghitung metrik keseluruhan pada set validasi. 
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).numpy()
    #mrnghitung metrik akhir 
    final_metrics = calculate_metrics(y_true, y_pred, num_classes=info.get('num_classes'))

    # save experiment info, config and results
    config_dict = {'args': vars(args), 'info': info}
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)

    results_dir = exp_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    metrics_out = {
        'best_val_accuracy': best_acc,
        'best_val_f1': best_f1,
        'final_val_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v.tolist()
                              for k, v in final_metrics.items() if k != 'confusion_matrix'},
        'train_history': {
            'loss': history['train_loss'],
            'acc': history['train_acc'],
            'f1': history.get('train_f1', [])
        },
        'val_history': {
            'loss': history['val_loss'],
            'acc': history['val_acc'],
            'f1': history['val_f1']
        }
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=4, default=str)

    # produce plots
    plot_learning_curves(
        {'loss': history['train_loss'], 'acc': history['train_acc']},
        {'loss': history['val_loss'], 'acc': history['val_acc']},
        save_path=results_dir / 'learning_curves.png'
    )
    plot_confusion_matrix(
        final_metrics['confusion_matrix'],
        info.get('class_names', []),
        save_path=results_dir / 'confusion_matrix.png'
    )
    plot_per_class_metrics(
        final_metrics,
        info.get('class_names', []),
        save_path=results_dir / 'per_class_metrics.png'
    )
    plot_roc_curves(
        y_true,
        y_probs,
        info.get('class_names', []),
        save_path=results_dir / 'roc_curves.png'
    )
    report = get_classification_report(y_true, y_pred, info.get('class_names', []))
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Final model saved to: {final_exp_path}")
    print(f"Best model saved to: {exp_ckpt_dir / 'best_model_ae_cn.pth'}")


#Program akan berjalan jika file ini dijalankan langsung.
if __name__ == '__main__':
    main()


"""
main()
│
├── [1. SETUP]
│   ├── argparse → baca argumen terminal
│   │   (--device, --ae_epochs, --clf_epochs, --batch_size, dll)
│   │
│   ├── torch.device()        → siapkan GPU/CPU
│   ├── Path()                → buat semua path folder
│   └── mkdir()               → buat folder experiments/, checkpoints/, logs/, results/
│
│
├── [2. LOAD DATA]  ← dari preprocessing.py dan dataset_old.py
│   │
│   ├── preprocessing.py
│   │   └── ECGPreprocessor()     → buat objek preprocessor
│   │           └── (belum proses sinyal, hanya inisialisasi)
│   │
│   └── dataset_old.py
│       └── create_dataloaders()
│               │
│               ├── ChapmanDataset(split='train')
│               │   ├── baca metadata.csv
│               │   ├── baca target_classes.csv
│               │   ├── _filter_target_classes()
│               │   │   └── _map_to_superclass()  ← tiap pasien
│               │   └── _create_splits()
│               │       ├── train_test_split() #1
│               │       └── train_test_split() #2
│               │
│               ├── ChapmanDataset(split='val')   ← proses sama
│               ├── ChapmanDataset(split='test')  ← proses sama
│               │
│               ├── get_class_distribution()      ← hitung jumlah tiap kelas
│               │
│               ├── [use_class_sampler=True]
│               │   ├── hitung class_weights = 1/class_counts
│               │   ├── loop tiap sampel → _map_to_superclass()
│               │   ├── buat sample_weights[]
│               │   ├── WeightedRandomSampler()
│               │   └── simpan class_weights.npy
│               │
│               └── DataLoader() × 3  → return train/val/test loader
│
│
├── [3. BUAT MODEL]  ← dari gru_cn_ae.py
│   ├── GRU_Autoencoder(input_size=1, hidden_size=128, num_layers=1)
│   └── GRU_FeatureClassifier(input_size=1, hidden_size=128, num_classes=4)
│
│
├── [4. PHASE 1: train_ae()]
│   │
│   ├── ae.to(device)
│   ├── Adam optimizer
│   ├── ReduceLROnPlateau scheduler  (mode='min', patience=2)
│   ├── MSELoss criterion
│   │
│   └── for epoch in range(ae_epochs):        ← default 10 epoch
│           └── for x, _ in train_loader:     ← label diabaikan
│                   │
│                   ├── __getitem__() dipanggil ← dataset_old.py
│                   │   ├── wfdb.rdrecord()    ← baca file ECG .hea/.mat
│                   │   ├── preprocessor.process()  ← preprocessing.py
│                   │   │   (filter, resample, normalize)
│                   │   └── return signal_tensor, label_tensor
│                   │
│                   ├── ae(x) → recon, latent  ← gru_cn_ae.py
│                   ├── MSELoss(recon, x)
│                   ├── backward()
│                   ├── clip_grad_norm_(max=1.0)
│                   └── opt.step()
│           │
│           ├── scheduler.step(avg_loss)
│           └── early stopping check (patience=3)
│
│
├── [5. PHASE 2: TRANSFER WEIGHTS]
│   ├── ae.encoder.state_dict()   → ambil bobot encoder AE
│   ├── clf.encoder.state_dict()  → ambil bobot encoder CLF
│   ├── loop → salin bobot yang cocok (shape sama)
│   └── clf.encoder.load_state_dict()  → masukkan bobot
│
│
├── [6. PHASE 3: train_classifier()]
│   │
│   ├── clf.to(device)
│   ├── Adam optimizer (lr lebih kecil: 5e-4, weight_decay=1e-5)
│   ├── ReduceLROnPlateau scheduler  (mode='max', patience=3)
│   ├── CrossEntropyLoss criterion
│   │
│   └── for epoch in range(clf_epochs):       ← default 30 epoch
│           │
│           ├── [TRAINING]
│           │   └── for x, y in train_loader:
│           │           ├── __getitem__() ← sama seperti phase 1
│           │           ├── clf(x) → logits   ← gru_cn_ae.py
│           │           ├── CrossEntropyLoss(logits, y)
│           │           ├── backward()
│           │           ├── clip_grad_norm_(max=5.0)
│           │           └── opt.step()
│           │
│           ├── [VALIDASI]
│           │   └── for x, y in val_loader:
│           │           ├── clf(x) → logits
│           │           └── argmax → preds
│           │
│           ├── calculate_metrics()  ← metrics.py
│           │   (accuracy, f1_macro)
│           │
│           ├── scheduler.step(val_acc)
│           │
│           ├── jika val_acc membaik → torch.save(best_model)
│           └── early stopping check (patience=5)
│
│
└── [7. EVALUASI AKHIR & SIMPAN]
        │
        ├── clf.eval()
        ├── loop val_loader → kumpulkan preds + probs
        │
        ├── calculate_metrics()       ← metrics.py
        │   (accuracy, f1, confusion matrix, per-class metrics)
        │
        ├── torch.save(final model)   → gru_ae_cn_improved.pth
        ├── shutil.copy()             → duplikat ke checkpoints/
        ├── json.dump(config)         → config.json
        ├── json.dump(metrics)        → metrics.json
        │
        ├── visualization.py
        │   ├── plot_learning_curves()    → learning_curves.png
        │   ├── plot_confusion_matrix()   → confusion_matrix.png
        │   ├── plot_per_class_metrics()  → per_class_metrics.png
        │   └── plot_roc_curves()         → roc_curves.png
        │
        └── get_classification_report()  ← metrics.py
            └── simpan classification_report.txt
"""