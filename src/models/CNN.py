import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        token = self.lang.getIndex(self.data[ind]['token'])
        last_token = self.lang.getIndex(self.data[ind]['last_token'])
        user = self.lang.getUserIndex(self.data[ind]['user'])
        x = torch.LongTensor([token, last_token, user]).unsqueeze(0)
        if self.labels == None: 
            y = self.data[ind]['instance_id']
        else:
            y = self.labels[ind]
        return x, y
  
def _collate(seq_list):
    return torch.cat([s[0] for s in seq_list], dim=0), [s[1] for s in seq_list]

# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    dataset = SLAMDataset(feats, labels, lang)
    dataloader = DataLoader(dataset, shuffle=(labels != None), batch_size=128, num_workers=2, collate_fn=_collate)
    return dataloader

class CNN(nn.Module):
    def __init__(self, vocab_size, num_users, embed_size, hidden_size, num_feats=3, dropout=0.2):
        super(CNN, self).__init__()
        self.vocab_embedding = nn.Embedding(vocab_size, embed_size)
        self.user_embedding = nn.Embedding(num_users, embed_size)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv1d(num_feats, hidden_size, 1, bias=False)
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: [token, last_token, user]
        tokens = self.vocab_embedding(x[:, :2])
        users = self.user_embedding(x[:, 2]).unsqueeze(1)
        x = torch.cat([tokens, users], dim=1)

        x = self.conv1(x)
        x = self.dropout(x)
        
        x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = self.out(x)
        return x

class Model:
    def __init__(self, lang):
        embed_size = 512
        hidden_size = 1024
        self.model = CNN(lang.num_words, lang.num_users, embed_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self, epoch):
        save_dir = './saved_model/'
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir + 'cnn_%d' % epoch)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.train()
        
        for epoch in range(epochs):
            losses = []
            start_time = time.time()
            for (feats, labels) in dataloader:
                self.optimizer.zero_grad()
                
                feats = feats.to(device)
                labels = torch.LongTensor(labels).to(device)
                outputs = self.model(feats)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                torch.cuda.empty_cache()

            end_time = time.time()
            print ('Epoch %d/%d | Loss %0.6f | Time %0.2fm' % (epoch+1, epochs, np.mean(np.array(losses)), (end_time-start_time)/60))
            self.save_model(epoch+1)

    def predict_for_set(self, dataloader, from_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if from_path != None:
            self.load_model(from_path)
        self.model.to(device)
        self.model.eval()

        predictions = dict()
        with torch.no_grad():
            for (feats, instance_ids) in dataloader:
                feats = feats.to(device)
                outputs = F.softmax(self.model(feats), dim=1)[:, 1]
                for i in range(len(outputs)):
                    predictions[instance_ids[i]] = outputs[i].item()
        return predictions
