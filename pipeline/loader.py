import pandas as pd
from hyperparams import *
from torch.utils.data import Dataset
import torch

class WindowLoader(Dataset):
    def __init__(self, df : pd.DataFrame, task):
        df = df.reset_index(drop=True)
        self.data = torch.tensor(df.values, dtype=torch.float32)
        if torch.cuda.is_available():
            self.data = self.data.cuda()

        if task == 'classifier':
            self.size = len(self.data) - WINDOW_SIZE
            
        elif task == 'regressor': 
            mask_indices = df[df['is_mask'] == 1].index.tolist()
            self.valid_indices = [i for i in mask_indices if i >= WINDOW_SIZE]
            self.size = len(self.valid_indices)

        features = {name: index for index, name in enumerate(df.columns)}
        self.feature_is = torch.tensor([
            features['sin_time'], features['cos_time'],
            features['volume_diff'], features['open_diff'],
            features['high_diff'], features['low_diff'],
            features['close_diff'], 
            features['open_r'], features['volume_r'], 
            features['velocity'], features['acceleration']
        ], device="cuda" if torch.cuda.is_available() else "cpu")

        self.features = features
        self.task = task
        
        def label_c(data, end):
            return data[end:end+1, 5]
        def label_r(data, end):
            return data[end, self.features['high_diff_asinh'] : self.features['close_diff_asinh'] + 1]
        
        self.get_label = label_c if task == 'classifier' else label_r

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        if self.task == 'classifier':
            start = index
            end = start + WINDOW_SIZE

            window = self.data[start : end]
            label = self.get_label(self.data, end)
            
        else:
            target_idx = self.valid_indices[index]
            
            start = target_idx - WINDOW_SIZE
            end = target_idx
            
            window = self.data[start : end]
            label = self.get_label(self.data, target_idx)

        window_transformed = torch.cat((
            (window[:, 0 : 4] - window[:, 0 : 4][0]) / window[:, 0 : 4][0],
            window.index_select(1, self.feature_is),
        ), dim=1)

        return window_transformed, label
    
