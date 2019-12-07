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
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list]) # (input_seq_len, batch_size)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list]) # (label_seq_len, batch_size)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    return token_input, token_len, label_input, label_len

def collate_test(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list]) # (seq_len, batch_size)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list]) # (label_seq_len, batch_size)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    ids = [s[2] for s in seq_list]
    return token_input, token_len, label_input, label_len, ids

# maximum prompt length 7
def get_dataloader(feats, lang, labels=None):
    batch_size = 64
    dataset = SLAMDataset(feats, labels, lang)
    print('dataset len', len(dataset))
    if labels is None:
        dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4, collate_fn=collate_test)
    else:
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4, collate_fn=collate_train)

    return dataloader
    
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=512):
        super(TransformerModel, self).__init__()
        self.output_embedding = nn.Embedding(2, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size)
        self.out = nn.Linear(embed_size, 2)

    def forward(self, x, x_len, y=None, y_len=None):
        x_mask = torch.arange(x.size(0)).unsqueeze(0).to(device) >= x_len.unsqueeze(1)
        y_mask = torch.arange(y.size(0)).unsqueeze(0).to(device) >= y_len.unsqueeze(1)
        x = self.embedding(x)
        y = self.output_embedding(y)
        output = self.transformer(
            x, # src (input_seq_len, batch_size, embed_size)
            y, # src (label_seq_len, batch_size, embed_size)
            src_key_padding_mask = x_mask, # (batch_size, input_seq_len)
            tgt_key_padding_mask = y_mask # (batch_size, label_seq_len)
        )
        output = self.out(output).transpose(0, 1)
        return output # (batch_size, label_seq_len, 2)

class Model:
    def __init__(self, lang):
        self.model = TransformerModel(len(lang.letters))
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(reduce=None).to(device)

    def save_model(self, epoch):
        save_dir = './saved_model/'
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir + 'transformer_%d' % epoch)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def step(self, train, token_input, token_len, label_input, label_len):
        token_input = token_input.to(device)
        token_len   = token_len.to(device)
        label_input = label_input.to(device)
        label_len   = label_len.to(device)
        predictions = self.model(token_input, token_len, label_input, label_len)
        if train:
            label_input = label_input.transpose(0, 1)
            mask = torch.zeros(label_input.size()).to(device)
            for i in range(len(label_len)):
                mask[i,:label_len[i]] = 1
            mask = mask.view(-1).to(device)
            
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            label_input = label_input.contiguous().view(-1)

            loss = self.criterion(predictions, label_input)
            masked_loss = torch.sum(loss*mask)
            if train: masked_loss.backward()
            current_loss = float(masked_loss.item())/int(torch.sum(mask).item())
            del token_input; del token_len; del label_input; del label_len
            del predictions; del mask
            return current_loss
        else:
            return predictions

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
                loss = self.step(True, token_input, token_len, label_input, label_len)
                losses.append(loss)
                if (batch_num+1) % 1000 == 0:
                    end_time = time.time()
                    print('Epoch %2d | Batch %5d | Loss %0.6f | Time %0.2fm' % (epoch+1, batch_num+1, loss, (end_time-start_time)/60))

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

            end_time = time.time()
            print ('Epoch %d/%d | Loss %0.6f | Time %0.2fm' % (epoch+1, epochs, sum(losses)/len(losses), (end_time-start_time)/60))
            self.save_model(epoch+1)

    def predict_for_set(self, dataloader, from_path=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if from_path != None:
            self.load_model(from_path)
        self.model.to(device)
        self.model.eval()

        predictions = dict()
        with torch.no_grad():
            for (batch_num, collate_output) in enumerate(dataloader):
                token_input, token_len, label_input, label_len, ids = collate_output
                preds = self.step(False, token_input, token_len, label_input, label_len)
                for i in range(len(ids)):
                    instance_ids = ids[i]
                    outputs = F.softmax(preds[i, :len(instance_ids), :], dim=1)
                    for j in range(len(instance_ids)):
                        instance_id = instance_ids[j]
                        output = outputs[j][1]
                        predictions[instance_id] = output.item()
        return predictions
