import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import os
import gc


 


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
        tmp = []
        self.max_length = -1
        for path in self.data_path:
            tmp = []
            data_np = np.load(path)
            for i in range(data_np.shape[0]):
                if i in channels_epoc_plus:
                    tmp.append(data_np[i])
            tmp = torch.tensor(tmp, dtype=torch.float32)
            self.max_length = max(tmp.shape[1], self.max_length)
            del tmp
            gc.collect()
        print("ciao: ", self.max_length)
        
            
 
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
        #mask = np.ones_like(data, dtype=int)
        #mean = torch.mean(data, dim=1, keepdim=True)
        #std = torch.std(data, dim=1, keepdim=True)
        #data = (data - mean) / std
        #data = torch.nn.functional.normalize(data)
        min_vals = data.min(dim=1, keepdim=True)[0]
        max_vals = data.max(dim=1, keepdim=True)[0]
        eps = 1e-8
        data = (data-min_vals)/((max_vals-min_vals)+eps)
        length = data.shape[1]
        padding = self.max_length - length
        data = np.pad(data, ((0,0), (0, padding), (0, 0)), mode='constant', constant_values=0)
        data = torch.tensor(data, dtype=torch.float32)
        num_channels, seq_length, num_bands = data.shape
        mask = np.zeros((num_channels, self.max_length, num_bands), dtype=int)
        mask[:, :seq_length, :] = 1# Set mask for valid data to 1
        mask = torch.tensor(mask, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return data, label, mask