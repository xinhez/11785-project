import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class SLAMDataset(Dataset):
    def __init__(self, data, labels, lang):
        super().__init__()
        self.lang = lang
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, ind):
        x = ' '.join(self.data[ind]['token'])
        x = torch.LongTensor([self.lang.letter2Index['<sos>']]+[self.lang.letter2Index[c] for c in x]+[self.lang.letter2Index['<eos>']]) # array of index of character
        if self.labels is None:
            y = torch.zeros(len(self.data[ind]['token'])).long()
            ids = [instance_id for instance_id in self.data[ind]['instance_id']]
            return x, y, ids
        else:
            y = torch.LongTensor(self.labels[ind])
            return x, y
  
def collate_train(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (batch_size, seq_len)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    return token_input, token_len, label_input, label_len

def collate_test(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (seq_len, batch_size)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True) # (label_seq_len, batch_size)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    ids = [s[2] for s in seq_list]
    return token_input, token_len, label_input, label_len, ids

# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    batch_size = 2
    dataset = SLAMDataset(feats, labels, lang)
    print('dataset len', len(dataset))
    if labels is None:
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4, collate_fn=collate_test)
    else:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, collate_fn=collate_train)

    return dataloader

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden_size*2, 2)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.out(output)
        return output

class Model:
    def __init__(self, lang):
        embed_size = 256
        hidden_size = 512
        self.model = RNN(len(lang.letters), embed_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)

    def save_model(self, epoch):
        save_dir = './saved_model/'
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir + 'rnn_%d' % epoch)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def to(self, device):
        self.model.to(device)

    def train(self, dataloader, epochs):
        self.model.to(device)
        self.model.train()
        
        for epoch in range(epochs):
            losses = []
            start_time = time.time()
            for (batch_num, collate_output) in enumerate(dataloader):
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                
                token_input, token_len, label_input, label_len = collate_output
                token_input = token_input.to(device)
                toekn_len   = token_len.to(device)
                outputs = self.model(token_input, token_len).contiguous().view(-1, 2)
                print('[0]', outputs.shape, token_len)
                print('[1]', label_input.shape)
                labels = nn.utils.rnn.pad_sequence(label_input, batch_first=True, padding_value=2).view(-1).to(device)

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
        if from_path != None:
            self.load_model(from_path)
        self.model.to(device)
        self.model.eval()

        predictions = dict()
        with torch.no_grad():
            for (batch_num, collate_output) in enumerate(dataloader):
                token_input, token_len, label_input, label_len, ids = collate_output
                token_input = token_input.to(device)
                toekn_len   = token_len.to(device)
                preds = self.model(token_input, token_len)
                for i in range(len(ids)):
                    instance_ids = ids[i]
                    outputs = F.softmax(preds[i, :len(instance_ids), :], dim=1)
                    for j in range(len(instance_ids)):
                        instance_id = instance_ids[j]
                        output = outputs[j][1]
                        predictions[instance_id] = output.item()
        return predictions
