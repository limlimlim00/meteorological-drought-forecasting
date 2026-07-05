import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

# 연도 기준 분할: train 1991–2015 / val 2016–2020 / test 2021–2025
TRAIN_END = '2015-12-31'
VAL_END   = '2020-12-31'
ALL_FEATURES = ['SPI1','SPI2','SPI3','SPI4','SPI5','SPI6','SPI9','SPI12','SPI18','SPI24']


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)  # (N, seq_len, n_feat)
        self.y = torch.FloatTensor(y)  # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_splits(data_path: str, station: str, target: str,
                seq_len: int, feature_cols: list = None):
    """
    Load monthly SPI data and return train/val/test splits as numpy arrays.

    Returns
    -------
    splits : dict  {'train': (X, y, dates), 'val': ..., 'test': ...}
    scaler : fitted MinMaxScaler
    target_idx : column index of target in feature_cols
    feature_cols : list of feature names actually used
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    df = pd.read_excel(data_path, sheet_name=station)
    df['일시'] = pd.to_datetime(df['일시'])
    df = df.set_index('일시').sort_index()[feature_cols]

    target_idx = feature_cols.index(target)

    arr = df.values.astype(np.float32)

    # Fit scaler on train split only
    train_mask = df.index <= TRAIN_END
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(arr[train_mask])
    arr = scaler.transform(arr)

    # Sliding window
    X_all, y_all = [], []
    for i in range(seq_len, len(arr)):
        X_all.append(arr[i - seq_len:i])   # (seq_len, n_feat)
        y_all.append(arr[i, target_idx])

    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    # Index after windowing
    idx = df.index[seq_len:]
    masks = {
        'train': idx <= TRAIN_END,
        'val':   (idx > TRAIN_END) & (idx <= VAL_END),
        'test':  idx > VAL_END,
    }

    splits = {
        name: (X_all[m], y_all[m], idx[m])
        for name, m in masks.items()
    }

    return splits, scaler, target_idx, feature_cols


def make_loaders(splits: dict, batch_size: int):
    """Return DataLoaders for train/val/test."""
    loaders = {}
    for split, (X, y, _) in splits.items():
        ds = TimeSeriesDataset(X, y)
        shuffle = (split == 'train')
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loaders
