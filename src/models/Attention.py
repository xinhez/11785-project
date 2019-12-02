import numpy as np
import time
import torch
from torch.utils.data import Dataset, DataLoader

class SLAMDataset(Dataset):
    def __init__(self, data, labels, lang):
        super().__init__()
        self.lang = lang
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, ind):
        x = np.array([self.lang.getIndex(token) for token in self.data[ind]['token']])
        if self.labels == None:
            return x
        y = np.array(self.labels[ind])
        return x, y
        
# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    dataset = SLAMDataset(feats, labels, lang)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=8)
    return dataloader

class AttentionModel:
    pass

class Model:
    def __init__(self, lang):
        self.model = AttentionModel()

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for epoch in range(epochs):
            avg_loss = 0
            start_time = time.time()
            for (data, labels) in dataloader:
                pass
            end_time = time.time()
            print ('Epoch %d/%d | Loss %0.6f | Time %0.2fm' % (epoch+1, epochs, avg_loss, (end_time-start_time)/60))

    def predict_for_set(self, dataloader):
        pass