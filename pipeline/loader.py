import pandas as pd
import torch
from torch.utils.data import Dataset

import hyperparams as hp


class WindowLoader(Dataset):
    """Sliding window dataset for time-series transformer training.

    For the classifier task, windows are contiguous slices of the full dataset.
    For the regressor task, windows are anchored at trend-masked data points.
    """

    def __init__(self, df: pd.DataFrame, task: str):
        df = df.reset_index(drop=True)
        self.data = torch.tensor(df.values, dtype=torch.float32)
        if torch.cuda.is_available():
            self.data = self.data.cuda()

        if task == 'classifier':
            self.size = len(self.data) - hp.WINDOW_SIZE
        elif task == 'regressor':
            mask_indices = df[df['is_mask'] == 1].index.tolist()
            self.valid_indices = [i for i in mask_indices if i >= hp.WINDOW_SIZE]
            self.size = len(self.valid_indices)

        # Map column names to tensor indices for feature selection
        column_indices = {name: index for index, name in enumerate(df.columns)}
        self.feature_indices = torch.tensor([
            column_indices['sin_time'], column_indices['cos_time'],
            column_indices['volume_diff'], column_indices['open_diff'],
            column_indices['high_diff'], column_indices['low_diff'],
            column_indices['close_diff'],
            column_indices['open_r'], column_indices['volume_r'],
            column_indices['velocity'], column_indices['acceleration'],
        ], device="cuda" if torch.cuda.is_available() else "cpu")

        self.column_indices = column_indices
        self.task = task

        # Select the appropriate label extraction function
        if task == 'classifier':
            self._get_label = lambda data, end: data[end:end + 1, 5]
        else:
            asinh_start = self.column_indices['high_diff_asinh']
            asinh_end = self.column_indices['close_diff_asinh'] + 1
            self._get_label = lambda data, end: data[end, asinh_start:asinh_end]

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.task == 'classifier':
            start = index
            end = start + hp.WINDOW_SIZE
            window = self.data[start:end]
            label = self._get_label(self.data, end)
        else:
            target_idx = self.valid_indices[index]
            start = target_idx - hp.WINDOW_SIZE
            window = self.data[start:target_idx]
            label = self._get_label(self.data, target_idx)

        # Normalize OHLC columns relative to window start, then append derived features
        ohlc = window[:, 0:4]
        ohlc_normalized = (ohlc - ohlc[0]) / ohlc[0]
        derived_features = window.index_select(1, self.feature_indices)
        window_transformed = torch.cat((ohlc_normalized, derived_features), dim=1)

        return window_transformed, label
