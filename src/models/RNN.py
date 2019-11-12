import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import numpy as np

def pad_collate(batch):
    (dataX, dataY) = zip(*batch)
    X_lens = [len(x) for x in dataX]

    X = pad_sequence(dataX, batch_first=True, padding_value=0)

    return X, X_lens, Y

class SLAMDataset(Dataset):
    def __init__(self, data, labels, lang):
        super().__init__()
        self.data = data
        self.labels = labels
        self.lang = lang
    
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, index):
        x = np.array([self.lang.getIndex(instance.token) for instance in self.data[index]])
        try:
            y = np.array([self.labels[instance.instance_id] for instance in self.data[index]])
        except:
            y = [instance.instance_id for instance in self.data[index]]
        return x, y

def get_dataloader(feats, labels, lang):
    dataset = SLAMDataset(feats, labels, lang)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True, collate_fn=pad_collate)
    return dataloader

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        hidden_size = 8
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, X, lengths):
        X = self.embedding(X)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False, batch_first=True)
        print("\nShapes in packed embedding: \n\t", [px.shape for px in packed_X])
        packed_out = self.rnn(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out, batch_first=True)
        out = self.output(out)
        return out, out_lens

class Model:
    def __init__(self, lang):
        hidden_size = 8
        self.model = RNN(lang.num_words, 2)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.train()
        self.model.to(device)

        running_loss = 0.0
        
        for epoch in range(epochs):
            print('\tTraining epoch %d/%d' % (epoch+1, epochs), end='\t')
            start_time = time.time()
            for (X, X_lens, Y) in dataloader:
                self.optimizer.zero_grad()

                print('input length', X_lens)
                print('input size', X.size())
                
                out, out_lens = self.model(X, X_lens)
                print('output', out.size())
                print('output length', out_lens)
                del X; del X_lens

                for i in range(len(Y)):
                    outputs = out[:out_lens[i].item(), i, :]
                    print('outputs', outputs.size())
                    targets = torch.LongTensor(trainY[i]).to(device)
                    print('targets', targets.size())
                    loss = self.criterion(outputs, targets)
                    running_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

            
            end_time = time.time()
            
            running_loss /= len(dataloader)
            print('Processing Time: %0.2f min' % ((end_time - start_time)/60),end='\t')
            print('Training Loss: ', running_loss)

    def predict_for_set(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(device)

        predictions = dict()
        
        start_time = time.time()
        with torch.no_grad():
            for (data, ids) in dataloader:
                pass

        end_time = time.time()
        
        print('\t Time: %0.2f min' % ((end_time - start_time)/60))
        return predictions
    