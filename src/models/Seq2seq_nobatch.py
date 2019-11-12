import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def _collate(seq_list):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return [torch.from_numpy(s[0]) for s in seq_list], [s[1] for s in seq_list]

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
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=8, pin_memory=True, collate_fn=_collate)
    return dataloader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, output, hidden):
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class Model:
    def __init__(self, lang):
        hidden_size = 8
        self.encoder = Encoder(lang.num_words, hidden_size)
        self.decoder = Decoder(hidden_size ,2)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters())
        self.decoder_optimizer = optim.Adam(self.decoder.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder.train()
        self.encoder.to(device)
        self.decoder.train()
        self.decoder.to(device)

        running_loss = 0.0
        
        for epoch in range(epochs):
            print('\tTraining epoch %d/%d' % (epoch+1, epochs), end='\t')
            start_time = time.time()
            for (data, labels) in dataloader:
                self.encoder_optimizer.zero_grad() 
                self.decoder_optimizer.zero_grad()

                encoder_hidden = self.encoder.initHidden()

                for word in data[0]:
                    word = word.to(device)
                    encoder_output, encoder_hidden = self.encoder(word, encoder_hidden)

                decoder_input = torch.tensor([[[.5, .5]]], device=device)
                decoder_hidden = encoder_hidden
                
                outputs = []
                for _ in data[0]:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs.append(decoder_output.view(-1))
                    decoder_input = decoder_output.detach()

                labels = torch.tensor(labels[0], device=device).long()
                outputs = torch.stack(outputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
            
            end_time = time.time()
            
            running_loss /= len(dataloader)
            print('Processing Time: %0.2f min' % ((end_time - start_time)/60),end='\t')
            print('Training Loss: ', running_loss)

    def predict_for_set(self, dataloader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder.eval()
        self.encoder.to(device)
        self.decoder.eval()
        self.decoder.to(device)

        predictions = dict()
        
        start_time = time.time()
        with torch.no_grad():
            for (data, ids) in dataloader:
                encoder_hidden = self.encoder.initHidden()

                for word in data[0]:
                    word = word.to(device)
                    encoder_output, encoder_hidden = self.encoder(word, encoder_hidden)

                decoder_input = torch.tensor([[[.5, .5]]], device=device)
                decoder_hidden = encoder_hidden
                
                outputs = []
                for instance_id in ids[0]:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    predictions[instance_id] = (F.softmax(decoder_output.view(-1), dim=0)[1])
                    decoder_input = decoder_output.detach()

        end_time = time.time()
        
        print('\t Time: %0.2f min' % ((end_time - start_time)/60))
        return predictions
    