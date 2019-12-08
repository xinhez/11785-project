import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SLAMDataset(Dataset):
    def __init__(self, data, labels, lang):
        super().__init__()
        self.lang = lang
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, ind):
        u = self.lang.getUserIndex(self.data[ind]['user'])
        x = ' '.join(self.data[ind]['token'])
        x = torch.LongTensor([self.lang.letter2Index['<sos>']]+[self.lang.letter2Index[c] for c in x]+[self.lang.letter2Index['<eos>']]) # array of index of character
        if self.labels is None:
            y = torch.zeros(len(self.data[ind]['token'])).long()
            ids = [instance_id for instance_id in self.data[ind]['instance_id']]
            return x, y, ids, u
        else:
            y = torch.LongTensor(self.labels[ind])
            return x, y, u
  
def collate_train(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (batch_size, seq_len)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True, padding_value=2)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    users = [s[2] for s in seq_list]
    return token_input, token_len, label_input, label_len, users

def collate_test(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (seq_len, batch_size)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True, padding_value=2) # (label_seq_len, batch_size)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    ids = [s[2] for s in seq_list]
    users = [s[3] for s in seq_list]
    return token_input, token_len, label_input, label_len, ids, users

# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    batch_size = 256
    dataset = SLAMDataset(feats, labels, lang)
    print('dataset len', len(dataset))
    if labels is None:
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_test)
    else:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_train)

    return dataloader

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size 
        self.rnn = nn.LSTM(embed_size, self.hidden_size, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        _, state = self.rnn(x)
        return state

class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden_size*2, embed_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        return output, hidden

class Grader(nn.Module):
    def __init__(self, input_size):
        super(Grader, self).__init__()
        self.grade = nn.Linear(input_size, 2)

    def forward(self, outputs):
        return self.grade(outputs)

class Seq2seq(nn.Module):
    def __init__(self, vocab_size, user_size, embed_size, hidden_size):
        super(Seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.user_embedding = nn.Embedding(user_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(hidden_size, embed_size)
        self.grader = Grader(embed_size)

    def forward(self, x, x_len, seq_length, users):
        x = self.embedding(x) # shape (batch_size, seq_length) -> (batch_size, seq_length, embed_size)
        batch_size = x.size(0)
        u = self.user_embedding(users).unsqueeze(1)

        #######################################################################
        # Encoder
        encoder_state = self.encoder(x, x_len)
        
        #######################################################################
        # Decoder
        decoder_input = self.embedding(torch.LongTensor(np.random.choice(self.vocab_size, (batch_size, 1))).to(device)) # (batch_size, 1, embed_size)
        decoder_state = encoder_state

        outputs = torch.zeros(seq_length, batch_size, 2).to(device)
        for t in range(seq_length):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            outputs[t] = self.grader((decoder_output+u).squeeze(1))
            decoder_input = decoder_output
            
        outputs = outputs.transpose(0, 1) # (batch_size, seq_length, 2)

        return outputs

class Model:
    def __init__(self, lang):
        embed_size = 256
        hidden_size = 512
        self.model = Seq2seq(lang.num_words, lang.num_users, embed_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)

    def save_model(self, epoch):
        save_dir = './saved_model/'
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir + 'seq2seq_c_%d' % epoch)

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
                
                token_input, token_len, label_input, label_len, users = collate_output
                token_input = token_input.to(device)
                token_len = token_len.to(device)
                seq_len = label_input.size(1)
                users = torch.LongTensor(users).to(device)

                outputs = self.model(token_input, token_len, seq_len, users).contiguous().view(-1, 2) # (batch_size * seq_length, 2)
                labels = nn.utils.rnn.pad_sequence(label_input, batch_first=True, padding_value=2).view(-1).to(device)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()
                if (batch_num+1) % 100 == 0:
                    end_time = time.time()
                    print('Batch %5d | Time %0.2fm' % (batch_num+1, (end_time-start_time)/60))

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
            for token_input, token_len, label_input, label_len, ids, users in dataloader:
                token_input = token_input.to(device)
                token_len = token_len.to(device)
                seq_len = label_input.size(1)
                users = torch.LongTensor(users).to(device)
                
                outputs = F.softmax(self.model(token_input, token_len, seq_len, users), dim=2)[:, :, 1] # (batch_size, seq_length, 2)\
                for batch_num in range(token_input.size(0)):
                    seq_length = label_len[batch_num]
                    instance_ids = ids[batch_num]
                    for i in range(seq_length):
                        instance_id = instance_ids[i]
                        prediction = outputs[batch_num][i]
                        predictions[instance_id] = prediction.item()
        return predictions
