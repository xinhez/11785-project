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
        u = self.lang.getUserIndex(self.data[ind]['user'])
        x = torch.LongTensor([self.lang.getIndex(token) for token in self.data[ind]['token']])
        if self.labels == None:
            y = [instance_id for instance_id in self.data[ind]['instance_id']]
        else:
            y = torch.LongTensor(self.labels[ind])
        return x, y, u
  
def _collate(seq_list):
    return [s[0] for s in seq_list], [s[1] for s in seq_list], [s[2] for s in seq_list]

# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    dataset = SLAMDataset(feats, labels, lang)
    dataloader = DataLoader(dataset, shuffle=(labels != None), batch_size=256, num_workers=4, collate_fn=_collate)
    return dataloader

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size 
        self.rnn = nn.RNN(embed_size, self.hidden_size, num_layers=4, batch_first=True, bidirectional=True)

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        _, state = self.rnn(x)
        return state

class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super(Decoder, self).__init__()
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers=4, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_size*2, embed_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        return output, hidden

class Grader(nn.Module):
    def __init__(self, embed_size, user_size):
        super(Grader, self).__init__()
        hidden = 1024
        self.user_embedding = nn.Embedding(user_size, embed_size)
        self.layer1 = nn.Linear(embed_size*2, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.layer3 = nn.Linear(hidden, hidden)
        self.grade = nn.Linear(hidden, 2)

    def forward(self, outputs, users):
        users = self.user_embedding(users)
        outputs = torch.cat([outputs, users], dim=1)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        return self.grade(outputs)

class Seq2seq(nn.Module):
    def __init__(self, vocab_size, user_size, embed_size, hidden_size):
        super(Seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(hidden_size, embed_size)
        self.grader = Grader(embed_size, user_size)

    def forward(self, x, x_len, users):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x = self.embedding(x) # shape (batch_size, seq_length) -> (batch_size, seq_length, embed_size)
        batch_size = x.size(0)
        seq_length = x.size(1)

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
            outputs[t] = self.grader(decoder_output.squeeze(1), users)
            decoder_input = decoder_output
            
        outputs = outputs.transpose(0, 1) # (batch_size, seq_length, 2)

        #######################################################################
        # Mask the outputs by actual length
        # for i in range(batch_size):
        #     if x_len[i].item() < seq_length:
        #         outputs[i, x_len[i].item():, :] = 0
        return outputs

class Model:
    def __init__(self, lang):
        embed_size = 512
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
        }, save_dir + 'seq2seq_usergrading_%d' % epoch)

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
            for (data, labels, users) in dataloader:
                self.optimizer.zero_grad()

                x_len = torch.LongTensor([len(seq) for seq in data]).to(device)
                x = nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)
                users = torch.LongTensor(users).to(device)

                outputs = self.model(x, x_len, users).contiguous().view(-1, 2) # (batch_size * seq_length, 2)
                labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=2).view(-1).to(device)
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
            for (data, instance_ids, users) in dataloader:
                x_len = torch.LongTensor([len(seq) for seq in data]).to(device)
                x = nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)
                users = torch.LongTensor(users).to(device)

                outputs = F.softmax(self.model(x, x_len, users), dim=2)[:, :, 1] # (batch_size, seq_length, 2)
                for batch_num in range(outputs.size(0)):
                    seq_length = x_len[batch_num]
                    for i in range(seq_length):
                        instance_id = instance_ids[batch_num][i]
                        prediction = outputs[batch_num][i]
                        predictions[instance_id] = prediction.item()
        return predictions
