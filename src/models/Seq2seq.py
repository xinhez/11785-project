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
    dataloader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=8, pin_memory=True, collate_fn=_collate)
    return dataloader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden, input_lengths):
        inputs = self.embedding(inputs)
        X = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True, enforce_sorted=False)
        output, state = self.gru(X, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, state

    def initHidden(self, batch_size):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

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
            for (trainX, trainY) in dataloader:
                self.encoder_optimizer.zero_grad() 
                self.decoder_optimizer.zero_grad()

                # X shape [max_length, batch_size]
                X_lens = torch.LongTensor([len(seq) for seq in trainX]).to(device)
                X = nn.utils.rnn.pad_sequence(trainX).to(device)
                del trainX

                encoder_output, encoder_state = self.encoder(X, X_lens)
                ... = self.decoder(encoder_output, encoder_state)

            
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
    