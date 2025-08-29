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
        # print(f"Loading episode from {self.file_list[idx]}")
        try:
            with open(self.file_list[idx], 'rb') as f:
                episode = pkl.load(f)
        except Exception as e:
            print(f"WARNING: Could not load {self.file_list[idx]}: {e}")
            # Skip to next file
            return self.__getitem__((idx+1)%len(self.file_list))
        # Optionally convert arrays to torch tensors
        # obs = episode['observations']
        # actions = episode['actions']
        rewards = episode['rewards']
        images = episode['images']
        # next_images = episode['next images']
        traj_len = episode['traj_len']
        # sa_t = np.concatenate([obs, actions], axis=-1)
        return images, rewards