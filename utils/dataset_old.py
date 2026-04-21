"""
PyTorch Dataset for Chapman-Shaoxing ECG Database
tugas : 
    - Load siyal
    - Panggil preprocessing
    - Encode Label
    - Siap dipakai DataLoader
"""

import os #mengatur path file
from unittest import signals
import numpy as np #library matematika 
import pandas as pd
import torch #library utama PyTorch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
#Dataset → class untuk membuat dataset custom
#DataLoader → mengambil batch data
#WeightedRandomSampler → mengatasi data imbalance
import wfdb #baca ecg database wfdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle #meyimpan split dataset

#Class Dataset (hanya utk penelitian ini)
class ChapmanDataset(Dataset):
    """
    PyTorch Dataset for Chapman-Shaoxing ECG Arrhythmia Classification
    
    Features:
    - On-the-fly preprocessing
    - Support single or multi-lead ECG
    - Multi-label support (one patient can have multiple diagnoses)
    - Train/Val/Test split
    """
    
    #fungsi ini dijalankan saat dataset dibuat, jalan sekali untuk inisialisai
    def __init__(self, 
                 data_dir, #folder ecg
                 metadata_path, #info pasien 
                 target_classes_path,#kelas target
                 preprocessor, #pakai preprocessing.py karna dipanggil train_gru_ae_cn_gn.py
                 split='train',
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 selected_lead=1, #Lead 2 saja yang digunakan
                 random_seed=42, #agar hasil split selalu sama
                 multi_label=False, #1 pasien hanya bisa punya 1 lbel
                 superclass_mode=False,
                 condition_names_path=None,
                 use_augmentation=False,
                 augmentation_prob=0.5,
                 augmentation_type='mixed'):
        """
        Args:
            data_dir: Path to WFDBRecords folder
            metadata_path: Path to metadata CSV
            target_classes_path: Path to target_classes CSV
            preprocessor: ECGPreprocessor instance
            split: 'train', 'val', or 'test'
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            selected_lead: Which lead to use (0-11, or 'all' for 12-lead)
            random_seed: Random seed for reproducibility
            multi_label: If True, handle multi-label classification
            use_augmentation: Whether to use augmentations for minority classes
            augmentation_prob: Probability of applying augmentation (0.0-1.0)
            augmentation_type: 'noise', 'crop', 'warp', or 'mixed'
        """
        #menyimpan parameter ke objek
        self.data_dir = data_dir
        self.preprocessor = preprocessor
        self.split = split

        # normalisasi selected_lead input
        if selected_lead != 'all':
            try:
                self.selected_lead = int(selected_lead) #ubah ke integer
            except Exception:
                raise ValueError(f"selected_lead must be integer 0-11 or 'all', got {selected_lead}")
        else:
            self.selected_lead = 'all'
        self.multi_label = multi_label
        self.use_augmentation = use_augmentation and (split == 'train')  # Only augment training data
        self.augmentation_prob = augmentation_prob
        self.augmentation_type = augmentation_type

        # Superclass mode: map all diagnoses to 4 superclasses (AFIB, GSVT, SB, SR)
        # If True, we will ignore target_classes_path filtering and map records by condition names
        self.superclass_mode = superclass_mode
        self.condition_names = {}
        self.condition_names_path = condition_names_path
        
        # baca file metadata 
        self.metadata = pd.read_csv(metadata_path)
        print(f"Loaded metadata: {len(self.metadata)} records")
        
        # Load target class (jika superclass_mode = false)
        self.target_classes = pd.read_csv(target_classes_path)
        self.target_codes = self.target_classes['snomed_code'].astype(str).tolist()
        self.class_names = self.target_classes['acronym'].tolist()
        self.num_classes = len(self.target_codes)

        # kalau superclass_mode = true, kita pakai 4 kelas utama
        if self.superclass_mode:
            self.class_names = ['AFIB', 'GSVT', 'SB', 'SR']
            self.num_classes = len(self.class_names)
        
        print(f"Target classes: {self.num_classes}")
        print(f"Classes: {self.class_names}")
        
        # Filter rekaman yang minimal punya 1 rekaman
        self.filtered_data = self._filter_target_classes()
        print(f"Records with target classes: {len(self.filtered_data)}")
        
        #  membagi dataset
        self.split_data = self._create_splits(train_ratio, val_ratio, test_ratio, random_seed)
        print(f"{split.upper()} set size: {len(self.split_data)}")
        
        # dipakai atau tidak : ...........?
        # Identify minority classes (for augmentation)
        if self.use_augmentation:
            self._identify_minority_classes()
        
    def _filter_target_classes(self):
        """
        Filter records that contain at least one target diagnosis
        """
        filtered_records = []
        
        #membaca setiap pasien
        for idx, row in self.metadata.iterrows():
            dx = row['dx'] #mengmbil diagnosis
            
            # Skip if no diagnosis
            if pd.isna(dx) or dx == 'NaN':
                continue
            
            # Parse diagnosis codes
            dx_codes = [code.strip() for code in str(dx).split(',')]

            if self.superclass_mode:
                # In superclass mode include any record with a diagnosis
                has_target = len(dx_codes) > 0
            else:
                # Check if any diagnosis is in target classes
                #tidak dipakai jika superclass_mode = true
                has_target = any(code in self.target_codes for code in dx_codes)

            if has_target:
                filtered_records.append({
                    'record_path': row['record_path'],
                    'dx_codes': dx_codes,
                    'age': row['age'],
                    'sex': row['sex']
                })
        
        return filtered_records
    
    def _create_splits(self, train_ratio, val_ratio, test_ratio, random_seed):
        """
        Create train/val/test splits with stratification
        """
        # Create primary labels for stratification
        # For normal mode: use the first matching target code
        # For superclass_mode: map dx_codes to one of the 4 superclasses
        primary_labels = []

        for record in self.filtered_data:
            dx_codes = record['dx_codes']
            if self.superclass_mode:
                # Map to superclass label name
                primary_labels.append(self._map_to_superclass(dx_codes))
            else:
                # Find first target class diagnosis
                for code in dx_codes:
                    if code in self.target_codes:
                        primary_labels.append(code)
                        break
        
        # Indices
        indices = np.arange(len(self.filtered_data))
        
        # membgi datset pertama : train+val vs test
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=primary_labels
        )
        
        train_labels = [primary_labels[i] for i in train_val_idx]
        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        
        #split kedua : train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_ratio,
            random_state=random_seed,
            stratify=train_labels
        )
        
        # Return corresponding split
        if self.split == 'train':
            split_indices = train_idx
        elif self.split == 'val':
            split_indices = val_idx
        else:  # test
            split_indices = test_idx
        
        return [self.filtered_data[i] for i in split_indices]
    
    #tidak dipakai jika use_augmentation = false, karena kita tidak perlu cari kelas minoritas untuk augmentasi
    def _identify_minority_classes(self):
        """
        Identify minority classes (below median) for augmentation
        """
        class_counts = np.zeros(self.num_classes, dtype=int)

        for record in self.split_data:
            dx_codes = record['dx_codes']
            if self.superclass_mode:
                lbl = self._map_to_superclass(dx_codes)
                class_idx = self.class_names.index(lbl)
                class_counts[class_idx] += 1
            else:
                for code in dx_codes:
                    if code in self.target_codes:
                        class_idx = self.target_codes.index(code)
                        class_counts[class_idx] += 1
        
        median_count = np.median(class_counts)
        self.minority_class_indices = set(np.where(class_counts < median_count)[0])
        print(f"Minority classes (count < {median_count:.0f}): {[self.class_names[i] for i in self.minority_class_indices]}")
    
    def _ensure_minority_classes_initialized(self):
        """
        Ensure minority_class_indices is initialized (even if augmentation is off)
        """
        if not hasattr(self, 'minority_class_indices'):
            self.minority_class_indices = set()  # Empty set if no augmentation
    
    #mengubah diagnosis menjadi angka
    def _encode_label(self, dx_codes):
        """
        Encode diagnosis codes to class indices
        
        Returns:
            - If multi_label=False: single class index (first matching)
            - If multi_label=True: binary vector (one-hot)
        """
        if self.superclass_mode:
            # Map to one of the 4 superclasses
            if self.multi_label:
                label_vector = np.zeros(self.num_classes, dtype=np.float32)
                lbl = self._map_to_superclass(dx_codes)
                idx = self.class_names.index(lbl)
                label_vector[idx] = 1.0
                return label_vector
            else:
                lbl = self._map_to_superclass(dx_codes)
                return self.class_names.index(lbl)

        #tidak dipakai jika superclass_mode = true, karena kita sudah map ke 4 kelas utama]
        if self.multi_label:
            # Multi-label: binary vector
            label_vector = np.zeros(self.num_classes, dtype=np.float32)
            
            for code in dx_codes:
                if code in self.target_codes:
                    class_idx = self.target_codes.index(code)
                    label_vector[class_idx] = 1.0
            
            return label_vector
        
        else:
            # Single-label: first matching diagnosis
            #tidak dipakai jika superclass_mode = true
            for code in dx_codes:
                if code in self.target_codes:
                    class_idx = self.target_codes.index(code)
                    return class_idx
            
            # Should not reach here due to filtering
            return 0

    def _load_condition_names(self, condition_names_path):
        """Load condition name map from CSV: snomed -> full name"""
        cond_map = {}
        try:
            import csv
            with open(condition_names_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row.get('Snomed_CT') or row.get('Snomed_CT ')
                    name = row.get('Full Name') or row.get('FullName') or row.get('Full Name ')
                    if code and name:
                        cond_map[code.strip()] = name.strip()
        except Exception:
            pass
        return cond_map

    def _map_to_superclass(self, dx_codes):
        """Map a list of diagnosis codes to one of the 4 superclasses.
        Heuristic based on condition name keywords.
        """
        # Lazy-load condition names if not present
        if not self.condition_names:
            # default path relative to repo
            default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'ConditionNames_SNOMED-CT.csv')
            self.condition_names = self._load_condition_names(default_path)

        afib_keys = ['fibril', 'atrial fibrillation', 'atrial flutter', 'afib', 'af']
        sr_keys = ['sinus rhythm', 'sr']
        sb_keys = ['sinus bradycardia', 'sb']
        tachy_keys = ['tachy', 'svt', 'ventricular', 'atrial tachy', 'reentrant', 'preexcitation']

        # First check for AFIB
        for c in dx_codes:
            name = self.condition_names.get(c, '').lower()
            if any(k in name for k in afib_keys):
                return 'AFIB'

        # Then SR
        for c in dx_codes:
            name = self.condition_names.get(c, '').lower()
            if any(k in name for k in sr_keys):
                return 'SR'

        # Then SB
        for c in dx_codes:
            name = self.condition_names.get(c, '').lower()
            if any(k in name for k in sb_keys):
                return 'SB'

        # Then tachy / ventricular -> GSVT
        for c in dx_codes:
            name = self.condition_names.get(c, '').lower()
            if any(k in name for k in tachy_keys):
                return 'GSVT'

        # Default
        return 'GSVT'
    
    #mengembalikan jumat data 
    def __len__(self):
        return len(self.split_data)
    
    #digunakan saaat trainign model, untuk mengambil satu sample data, jalan ribuan kali selama training
    def __getitem__(self, idx):
        """
        Load and preprocess ECG signal
        
        Returns:
            signal: Tensor of shape [1, signal_length] for single lead
                    or [12, signal_length] for 12-lead
            label: Class index (int) or binary vector (multi-label)
        """
        # Ensure minority_class_indices is initialized
        self._ensure_minority_classes_initialized()
        
        # Get record info
        record_info = self.split_data[idx]
        record_path = record_info['record_path']
        dx_codes = record_info['dx_codes']
        
        # Encode label to determine if minority class
        label = self._encode_label(dx_codes)
        is_minority = (not self.multi_label) and (label in self.minority_class_indices)
        
        # Load ECG record
        full_path = os.path.join(self.data_dir, record_path)
        
        try:
            record = wfdb.rdrecord(full_path)
            
            # Select lead(s)
            if self.selected_lead == 'all':
                # Use all 12 leads
                ecg_signal = record.p_signal  # Shape: [length, 12]
                
                # Preprocess each lead
                processed_leads = []
                for lead_idx in range(12):
                    lead_signal = ecg_signal[:, lead_idx]
                    processed = self.preprocessor.process(lead_signal)
                    
                    #tidak jalan jika use_augmentation = false, karena kita tidak akan augmentasi
                    if self.use_augmentation and is_minority and np.random.rand() < self.augmentation_prob:
                        processed = self.preprocessor.augment_signal(processed, self.augmentation_type)
                    
                    processed_leads.append(processed)
                
                # Stack: [12, signal_length]
                processed_signal = np.stack(processed_leads, axis=0)
                
            else:
                # Use single lead
                ecg_signal = record.p_signal[:, self.selected_lead]
                
                # Preprocess
                processed_signal = self.preprocessor.process(ecg_signal)
                
                # Apply augmentation to minority classes during training
                if self.use_augmentation and is_minority and np.random.rand() < self.augmentation_prob:
                    processed_signal = self.preprocessor.augment_signal(processed_signal, self.augmentation_type)
                
                # Add channel dimension: [1, signal_length]
                processed_signal = processed_signal[np.newaxis, :]
            
            # Convert to tensor
            signal_tensor = torch.FloatTensor(processed_signal)
            
            if self.multi_label:
                label_tensor = torch.FloatTensor(label)
            else:
                label_tensor = torch.LongTensor([label])
            
            return signal_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading {record_path}: {e}")
            # Return a dummy sample
            if self.selected_lead == 'all':
                dummy_signal = torch.zeros(12, self.preprocessor.target_length)
            else:
                dummy_signal = torch.zeros(1, self.preprocessor.target_length)
            
            if self.multi_label:
                dummy_label = torch.zeros(self.num_classes)
            else:
                dummy_label = torch.LongTensor([label if not self.multi_label else 0])
            
            return dummy_signal, dummy_label
    
    def get_class_distribution(self):
        """
        Get class distribution in current split
        """
        class_counts = np.zeros(self.num_classes, dtype=int)

        for record in self.split_data:
            dx_codes = record['dx_codes']
            if self.superclass_mode:
                lbl = self._map_to_superclass(dx_codes)
                try:
                    class_idx = self.class_names.index(lbl)
                    class_counts[class_idx] += 1
                except ValueError:
                    # unknown label -> skip
                    continue
            else:
                for code in dx_codes:
                    if code in self.target_codes:
                        class_idx = self.target_codes.index(code)
                        class_counts[class_idx] += 1
        
        # Create DataFrame
        distribution = pd.DataFrame({
            'class_name': self.class_names,
            'count': class_counts,
            'percentage': class_counts / len(self.split_data) * 100
        })
        
        return distribution
    
    def save_split_info(self, save_path):
        """
        Save train/val/test split info for reproducibility
        """
        split_info = {
            'split': self.split,
            'num_samples': len(self.split_data),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'records': [r['record_path'] for r in self.split_data]
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(split_info, f)
        
        print(f"Split info saved to {save_path}")


def create_dataloaders(data_dir,
                      metadata_path,
                      target_classes_path,
                      preprocessor,
                      superclass_mode=False,
                      condition_names_path=None,
                      batch_size=32,
                      num_workers=4,
                      pin_memory=True,
                      selected_lead=1,
                      random_seed=42,
                      use_class_sampler=False,
                      use_augmentation=False,
                      augmentation_prob=0.5,
                      augmentation_type='mixed'):
    """
    Create train/val/test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, dataset_info
    """
    
    # Create datasets
    train_dataset = ChapmanDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        split='train',
        selected_lead=selected_lead,
        random_seed=random_seed,
        use_augmentation=use_augmentation,
        augmentation_prob=augmentation_prob,
        augmentation_type=augmentation_type,
        superclass_mode=superclass_mode,
        condition_names_path=condition_names_path
    )
    
    val_dataset = ChapmanDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        split='val',
        selected_lead=selected_lead,
        random_seed=random_seed,
        use_augmentation=False  # No augmentation for val
        ,superclass_mode=superclass_mode,
        condition_names_path=condition_names_path
    )
    
    test_dataset = ChapmanDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        split='test',
        selected_lead=selected_lead,
        random_seed=random_seed,
        use_augmentation=False,  # No augmentation for test
        superclass_mode=superclass_mode,
        condition_names_path=condition_names_path
    )
    
    # Create dataloaders
    # Optionally create a WeightedRandomSampler to rebalance classes in each batch
    train_loader = None
    class_counts = train_dataset.get_class_distribution()['count'].values
    class_counts = np.array(class_counts, dtype=np.int64)

    if use_class_sampler:
        # Compute inverse-frequency weights per class
        #kelas yang dikit dapat bobot lebih besar
        eps = 1e-6
        class_weights = 1.0 / (class_counts + eps)

        # tiap kelas
        sample_weights = []
        for record in train_dataset.split_data:
            dx_codes = record['dx_codes']
            if superclass_mode:
                #dicari pasien ini masuk kelas apa berdasarkan diagnosisnya, lalu ambil bobot kelas itu
                lbl = train_dataset._map_to_superclass(dx_codes) #misal -> AFIB
                class_idx = train_dataset.class_names.index(lbl) # -> 0 (index AFIB)
            else:
                class_idx = None
                for code in dx_codes:
                    if code in train_dataset.target_codes:
                        class_idx = train_dataset.target_codes.index(code)
                        break
                if class_idx is None:
                    class_idx = 0
            # beri bobot sesuai kelasnya
            sample_weights.append(class_weights[class_idx])

        #sampel_weight = list sepanjang jumlah data trainign isinya bobot tiap sampel
        sample_weights = np.array(sample_weights, dtype=np.float64)

        # sampler memakai bobot untuk pilih batch
        sampler = WeightedRandomSampler(
            weights=sample_weights, #bobot tiap sample
            num_samples=len(sample_weights), #ambil sebanyak total data
            replacement=True ##boleh sampel yang sama muncul beberapa kali dalam satu epoch
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

        # Save class weights to processed dir (so training code can load them)
        try:
            from pathlib import Path
            processed_dir = Path(metadata_path).parent
            processed_dir.mkdir(parents=True, exist_ok=True)
            np.save(processed_dir / 'class_weights.npy', class_weights.astype(np.float32))
            print(f"Saved class_weights.npy to {processed_dir}")
        except Exception as e:
            print(f"Warning: failed to save class_weights.npy: {e}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Drop last incomplete batch
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Dataset info
    dataset_info = {
        'num_classes': train_dataset.num_classes,
        'class_names': train_dataset.class_names,
        'class_counts': class_counts.tolist(),
        'class_weights': (class_weights.tolist() if 'class_weights' in locals() else None),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'signal_length': preprocessor.target_length,
        'num_leads': 12 if selected_lead == 'all' else 1
    }
    
    print("\n" + "="*70)
    print("DATALOADER INFO")
    print("="*70)
    print(f"Train samples: {dataset_info['train_size']:,}")
    print(f"Val samples  : {dataset_info['val_size']:,}")
    print(f"Test samples : {dataset_info['test_size']:,}")
    print(f"Batch size   : {batch_size}")
    print(f"Num workers  : {num_workers}")
    print(f"Num classes  : {dataset_info['num_classes']}")
    print(f"Signal length: {dataset_info['signal_length']}")
    print(f"Num leads    : {dataset_info['num_leads']}")
    signals, labels = next(iter(train_loader))
    print(signals.shape)
    print("="*70)

    
    return train_loader, val_loader, test_loader, dataset_info


if __name__ == "__main__":
    # Test dataset
    from utils.preprocessing_old import ECGPreprocessor
    
    print("Testing ChapmanDataset...")
    
    data_dir = '../data/raw/WFDBRecords'
    metadata_path = '../data/processed/metadata_full.csv'
    target_classes_path = '../data/processed/target_classes.csv'
    
    preprocessor = ECGPreprocessor()
    
    # Create dataset
    dataset = ChapmanDataset(
        data_dir=data_dir,
        metadata_path=metadata_path,
        target_classes_path=target_classes_path,
        preprocessor=preprocessor,
        split='train',
        selected_lead=0
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")
    
    # Test loading one sample
    signal, label = dataset[0]
    print(f"\nSample signal shape: {signal.shape}")
    print(f"Sample label: {label.item()} ({dataset.class_names[label.item()]})")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    signals, labels = next(iter(loader))
    
    print(f"\nBatch signal shape: {signals.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    print("\n✓ Dataset test passed!")

"""
SEBELUM TRAINING:
__init__ → filter data → split data → buat sampler → buat DataLoader

SAAT TRAINING (tiap batch):
__getitem__ → baca ECG → preprocessing → encode label → kirim ke model
"""