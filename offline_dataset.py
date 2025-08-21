import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import numpy as np

class OfflineDataset(Dataset):
    def __init__(self, data_dir):

        self.file_list = sorted([
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith('.pkl') and fname.startswith('data_')
        ])
        assert len(self.file_list) > 0, f"No PKL files found in {data_dir}"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            episode = pkl.load(f)
        # Optionally convert arrays to torch tensors
        obs = episode['observations']
        actions = episode['actions']
        rewards = episode['rewards']
        sa_t = np.concatenate([obs, actions], axis=-1)
        return sa_t, rewards