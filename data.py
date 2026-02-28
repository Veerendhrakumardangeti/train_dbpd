import os
import ast
import numpy as np
import pandas as pd
import wfdb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ECGDataset(Dataset):
    """
    Standard PyTorch Dataset for PTB-XL.
    Loads signals on-demand from filesystem for memory efficiency.
    """
    def __init__(self, df, data_path, x_scaler=None, y_scaler=None):
        self.df = df.copy()
        self.data_path = data_path
        
        # Features (Metadata)
        self.X = build_features(self.df)
        if x_scaler:
            self.X = pd.DataFrame(x_scaler.transform(self.X), columns=self.X.columns)
            
        # Labels
        # Labels (Single-Label integers 0-4)
        # We compute this once at init
        self.y_indices = build_single_labels(self.df)
        self.Z = torch.tensor(self.y_indices, dtype=torch.long)
        
        self.y_scaler = y_scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Load Signal (on-demand)
        row = self.df.iloc[idx]
        # We use sampling_rate 100 files by default here
        file_path = os.path.join(self.data_path, row.filename_lr)
        signal, _ = wfdb.rdsamp(file_path)
        
        # 2. Preprocess Signal
        if self.y_scaler:
            signal = self.y_scaler.transform(signal.reshape(-1, signal.shape[-1])).reshape(signal.shape)
        
        # Reshape (1000, 12) -> (12, 1000, 1) to match Model02 expectations
        signal = signal.T  # (12, 1000)
        signal = np.expand_dims(signal, axis=-1)  # (12, 1000, 1)
        
        # 3. Features & Labels
        # 3. Features & Labels
        meta = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
        ecg = torch.tensor(signal, dtype=torch.float32)
        label = self.Z[idx]  # LongTensor scalar
        
        return meta, ecg, label, idx


def load_metadata(data_path):
    ecg_df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
    ecg_df.scp_codes = ecg_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    scp_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]
    return ecg_df, scp_df


def add_diagnostic_classes(ecg_df, scp_df):
    def diagnostic_class(scp):
        res = set()
        for k in scp.keys():
            if k in scp_df.index:
                res.add(scp_df.loc[k].diagnostic_class)
        return list(res)

    ecg_df['scp_classes'] = ecg_df.scp_codes.apply(diagnostic_class)
    return ecg_df


def build_features(ecg_df):
    X = pd.DataFrame(index=ecg_df.index)
    X['age'] = ecg_df.age.fillna(0)
    X['sex'] = ecg_df.sex.astype(float).fillna(0)
    X['height'] = ecg_df.height
    X.loc[X.height < 50, 'height'] = np.nan
    X['height'] = X.height.fillna(0)
    X['weight'] = ecg_df.weight.fillna(0)
    X['infarction_stadium1'] = ecg_df.infarction_stadium1.replace({
        'unknown': 0, 'Stadium I': 1, 'Stadium I-II': 2,
        'Stadium II': 3, 'Stadium II-III': 4, 'Stadium III': 5
    }).fillna(0)
    X['infarction_stadium2'] = ecg_df.infarction_stadium2.replace({
        'unknown': 0, 'Stadium I': 1, 'Stadium II': 2, 'Stadium III': 3
    }).fillna(0)
    X['pacemaker'] = (ecg_df.pacemaker == 'ja, pacemaker').astype(float)
    return X


def build_single_labels(ecg_df):
    """
    Priority Rule: MI > STTC > CD > HYP > NORM
    Returns a numpy array of integers [0..4]
    
    Mapping:
    NORM: 0
    MI: 1
    STTC: 2
    CD: 3
    HYP: 4
    """
    # Priority order (strings matching scp_classes)
    # MI=1, STTC=2, CD=3, HYP=4, NORM=0
    
    y = np.zeros(len(ecg_df), dtype=int) # Default 0 (NORM)
    
    for i, (idx, row) in enumerate(ecg_df.iterrows()):
        cats = row.scp_classes # list of strings
        
        # Priority Check
        if 'MI' in cats:
            val = 1
        elif 'STTC' in cats:
            val = 2
        elif 'CD' in cats:
            val = 3
        elif 'HYP' in cats:
            val = 4
        else:
            val = 0 # NORM or other
            
        y[i] = val
        
    return y.astype(np.int64)


def build_labels(ecg_df):
    Z = pd.DataFrame(0, index=ecg_df.index,
                     columns=['NORM', 'MI', 'STTC', 'CD', 'HYP'], dtype='int')
    for i in Z.index:
        for k in ecg_df.loc[i].scp_classes:
            Z.loc[i, k] = 1
    return Z


def get_scalers(df_train, data_path):
    """Pre-calculate scalers using training data."""
    # Metadata Scaler
    X_train = build_features(df_train)
    x_scaler = StandardScaler().fit(X_train)
    
    # For parity, we sample 1000 signals to fit the scaler
    print("Fitting signal scaler (sampling 1000)...")
    sample_signals = []
    for f in df_train.filename_lr.sample(min(1000, len(df_train))):
        s, _ = wfdb.rdsamp(os.path.join(data_path, f))
        sample_signals.append(s)
    sample_signals = np.concatenate(sample_signals, axis=0)
    y_scaler = StandardScaler().fit(sample_signals)
    
    return x_scaler, y_scaler


def load_ptbxl(data_path, sampling_rate, batch_size):
    """Main entry point for data loading."""
    ecg_df, scp_df = load_metadata(data_path)
    ecg_df = add_diagnostic_classes(ecg_df, scp_df)

    # Train/Val/Test Split
    df_train = ecg_df[ecg_df.strat_fold <= 8]
    df_val   = ecg_df[ecg_df.strat_fold == 9]
    df_test  = ecg_df[ecg_df.strat_fold == 10]

    # Calculate Scalers
    x_scaler, y_scaler = get_scalers(df_train, data_path)

    # Create Datasets
    train_ds = ECGDataset(df_train, data_path, x_scaler, y_scaler)
    val_ds   = ECGDataset(df_val,   data_path, x_scaler, y_scaler)
    test_ds  = ECGDataset(df_test,  data_path, x_scaler, y_scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Class distribution (for loss weighting)
    Z_train = build_single_labels(df_train)  # Get 1D array of single labels (0-4)
    cls_num_list = np.bincount(Z_train, minlength=5).astype(np.float64)  # Count occurrences of each class

    return (train_loader, val_loader, test_loader, cls_num_list)
