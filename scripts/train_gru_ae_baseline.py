"""
GRU Autoencoder + NO Normalization Training (Pure Baseline)
with proper scheduler, early stopping, and gradient clipping.

Similar to train_gru_ae_cn_gn.py but uses NO normalization (pure baseline).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import time
import os
import json
import datetime
import shutil

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import utils.dataset_old as dataset_old
from models.gru_baseline_ae import GRU_Autoencoder, GRU_FeatureClassifier
from utils.metrics import calculate_metrics, get_classification_report
from utils.preprocessing import ECGPreprocessor
from utils.visualization import (
    plot_learning_curves, 
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_curves
)

def train_ae(ae, loader, device, epochs=10, lr=1e-3):
    """Improved autoencoder training."""
    ae.to(device)
    ae.train()
    opt = optim.Adam(ae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', patience=2, factor=0.5
    )
    criterion = nn.MSELoss() 

    best_loss = float('inf')
    patience_counter = 0
    patience = 3

    for epoch in range(epochs):
        t0 = time.time()
        running = 0.0
        pbar = tqdm(loader, desc=f'AE Epoch {epoch+1}/{epochs}')
        
        for x, _ in pbar:
            x = x.to(device)
            recon, _ = ae(x)
            loss = criterion(recon, x)
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            
            opt.step()
            running += loss.item() * x.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running / len(loader.dataset)
        elapsed = time.time() - t0
        print(f"AE Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Time: {elapsed:.1f}s")
        
        # Scheduler step
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping AE at epoch {epoch+1}")
                break


def train_classifier(clf, train_loader, val_loader, device, epochs=30, lr=5e-4,
                     checkpoint_dir=None):
    """Improved classifier training with scheduler and early stopping.

    Args:
        checkpoint_dir: optional Path-like directory where the best model will be saved.
    """
    clf.to(device)
    opt = optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-5)
    
    # ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=3, factor=0.5
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 5  # Early stopping patience
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    for epoch in range(epochs):
        # ===== TRAINING PHASE =====
        clf.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device).view(-1)
            
            logits = clf(x)
            loss = criterion(logits, y)
            
            opt.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=5.0)
            
            opt.step()
            
            running_loss += loss.item() * x.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # ===== VALIDATION PHASE =====
        clf.eval()
        val_running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for x, y in pbar_val:
                x = x.to(device)
                y = y.to(device).view(-1)
                
                logits = clf(x)
                loss = criterion(logits, y)
                
                val_running_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
        
        val_loss = val_running_loss / len(val_loader.dataset)
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        metrics = calculate_metrics(y_true.numpy(), y_pred.numpy(), num_classes=None)
        
        val_acc = metrics['accuracy']
        val_f1 = metrics.get('f1_macro', metrics.get('f1', 0.0))
        
        # Store history
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
        
        # Scheduler step (based on validation accuracy)
        old_lr = opt.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  → LR reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  ✓ New best! Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f}")
            
            # Save best model (to experiment folder if provided)
            save_path = 'checkpoints/best_model_ae_baseline.pth'
            if checkpoint_dir is not None:
                save_path = str(Path(checkpoint_dir) / 'best_model_ae_baseline.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': clf.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_acc': best_val_acc,
                'val_f1': best_val_f1,
                'history': history
            }, save_path)
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
            break
        
        print("-" * 70)
    
    return best_val_acc, best_val_f1, history


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

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
    exp_name = f'gru_ae_baseline_{timestamp}'
    exp_dir = exp_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'results').mkdir(exist_ok=True)

    # also keep a copy in the global checkpoints folder for backward compatibility
    default_ckpt_dir = project_root / 'checkpoints'
    default_ckpt_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("GRU AUTOENCODER + NO NORMALIZATION (PURE BASELINE) TRAINING")
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

    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader, info = dataset_old.create_dataloaders(
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        superclass_mode=True,
        condition_names_path=condition_names_path,
        use_class_sampler=True,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        selected_lead=args.selected_lead
    )

    print(f"\n✓ Dataset loaded:")
    print(f"  Classes: {info['class_names']}")
    print(f"  Train: {info['train_size']:,} samples")
    print(f"  Val: {info['val_size']:,} samples\n")

    # Instantiate models
    ae = GRU_Autoencoder(
        input_size=info.get('num_leads', 1),
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
    train_ae(ae, train_loader, device, epochs=args.ae_epochs, lr=args.ae_lr)

    # ===== PHASE 2: TRANSFER WEIGHTS =====
    print("\n" + "="*70)
    print("PHASE 2: TRANSFERRING ENCODER WEIGHTS")
    print("="*70)
    ae_state = ae.encoder.state_dict()
    clf_state = clf.encoder.state_dict()
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
    best_acc, best_f1, history = train_classifier(
        clf, train_loader, val_loader, device,
        epochs=args.clf_epochs, lr=args.clf_lr,
        checkpoint_dir=exp_dir / 'checkpoints'
    )

    # ===== SAVE FINAL MODEL =====
    exp_ckpt_dir = exp_dir / 'checkpoints'
    exp_ckpt_dir.mkdir(exist_ok=True)

    final_exp_path = exp_ckpt_dir / 'gru_ae_baseline_final.pth'
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

    # perform a final evaluation on the validation set and save results
    clf.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).view(-1)
            logits = clf(x)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).numpy()
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
    print(f"Best model saved to: {exp_ckpt_dir / 'best_model_ae_baseline.pth'}")


if __name__ == '__main__':
    main()
