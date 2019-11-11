import time
import torch
import torch.nn as nn
import torch.optim as optim
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(1, 2))
        layers.append(nn.Softmax())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, 1]

class Model:
    def __init__(self):
        self.model = MLP()
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.L1Loss()

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.train()
        self.model.to(device)

        running_loss = 0.0
        
        for epoch in range(epochs):
            print('Training epoch %d/%d' % (epoch+1, epochs), end='\t')
            start_time = time.time()
            for (feats, labels) in dataloader:
                self.optimizer.zero_grad() 
                feats = feats.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float()

                outputs = self.model(feats)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            
            end_time = time.time()
            
            running_loss /= len(dataloader)
            print('Processing Time: %0.2f min' % ((end_time - start_time)/60))
            print('\tTraining Loss: ', running_loss)

    def predict_test_set(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(device)

        predictions = dict()
        
        start_time = time.time()
        with torch.no_grad():
            for (feats, ids) in dataloader:
                feats = feats.to(device, non_blocking=True).float()
                outputs = self.model(feats)
                for i in range(len(feats)):
                    predictions[ids[i]] = outputs[i].item()
        
        end_time = time.time()
        
        print('Inference Time: %0.2f min' % ((end_time - start_time)/60))
        return predictions
    