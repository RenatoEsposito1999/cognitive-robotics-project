import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os


 


channels_epoc_plus = [3,4,5,7,11,13,15,21,23,31,41,49,58,60]
 
class EEGDataset(Dataset):
    """
    Lazy-loading EEG dataset for each .mat file with 24 EEG sequences and corresponding labels.
    """
    def __init__(self, path, selected_channel):
        self.data_path= []
        self.labels = []
        self.selected_channel = selected_channel
        for root, _, files in os.walk(path):
            for file in files:
                if(file.endswith(".npy")):
                    file_path = os.path.join(root, file)
                    self.data_path.append(file_path)
                    parts = file_path.split("_")
                    label = parts[-1].split(".")[0]
                    self.labels.append(int(label))
        
            
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        path_idx = self.data_path[idx]
        data_np = np.load(path_idx)
        tmp = []
        for i in range(data_np.shape[0]):
            if i in channels_epoc_plus:
                tmp.append(data_np[i])
        data = torch.tensor(tmp, dtype=torch.float32)
        #data = torch.nn.functional.normalize(data)
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        data = (data - mean) / std
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return data, label