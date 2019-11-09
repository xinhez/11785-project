import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SLAMDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
      
    def __getitem__(self, index):
        isinstance_id = self.x[index].instance_id
        try:
            y = np.array([self.y[isinstance_id]])
        except:
            y = isinstance_id
        return np.array([sum(self.x[index].to_features().values())]), y

def get_dataloader(feats, labels):
    dataset = SLAMDataset(feats, labels)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=512, num_workers=8, pin_memory=True)
    return dataloader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layers = []
        layers.append(nn.Linear(1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)