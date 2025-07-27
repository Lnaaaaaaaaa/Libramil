import os
import h5py
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

class WSIDataset(Dataset):
    def __init__(self, indices, label_csv, h5_file_dir):
        self.h5_file_dir = h5_file_dir

        df = pd.read_csv(label_csv)
        self.name_label = df.set_index('name')['label'].to_dict()

        self.features = {}   
        self.indices = []     

        for name in tqdm(indices, desc="Loading WSI features and texts"):
            h5_path = os.path.join(self.h5_file_dir, name + '.h5')

            if not os.path.exists(h5_path):
                continue
            if name not in self.name_label:
                continue

            try:
                with h5py.File(h5_path, 'r') as h5:
                    self.features[name] = torch.tensor(h5['features'][:], dtype=torch.float32)
                self.indices.append(name)
            except Exception as e:
                print(f'[Error] Loading {name}: {e}')
                continue

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        name = self.indices[idx]
        label = self.name_label[name]
        features = self.features[name]           

        return features, label
   
def get_dataloader(data_split_json, data_csv, h5_file_dir, idx=0):
    with open(data_split_json, 'r') as fp:
        indices = json.load(fp)
        train_set = WSIDataset(indices[f'train_{idx}'], data_csv, h5_file_dir)
        valid_set = WSIDataset(indices[f'val_{idx}'], data_csv, h5_file_dir)
        test_set  = WSIDataset(indices[f'test_{idx}'], data_csv, h5_file_dir)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=1)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
  