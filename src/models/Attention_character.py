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
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (batch_size, seq_len)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True, padding_value=2)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    return token_input, token_len, label_input, label_len

def collate_test(seq_list):
    token_input = torch.nn.utils.rnn.pad_sequence([s[0] for s in seq_list], batch_first=True) # (seq_len, batch_size)
    token_len   = torch.LongTensor([len(s[0]) for s in seq_list])
    label_input = torch.nn.utils.rnn.pad_sequence([s[1] for s in seq_list], batch_first=True, padding_value=2) # (label_seq_len, batch_size)
    label_len   = torch.LongTensor([len(s[1]) for s in seq_list])
    ids = [s[2] for s in seq_list]
    return token_input, token_len, label_input, label_len, ids

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

class Attention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.view(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size 
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
        self.out = nn.Linear(hidden_size*2, embed_size)

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=False)
        output, state = self.rnn(x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.out(output)
        return output, state

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.rnn = nn.LSTM(embed_size*2, hidden_size, num_layers=4, batch_first=True, bidirectional=True, dropout=0.2)
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
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(embed_size, hidden_size)
        self.decoder = Decoder(embed_size, hidden_size)
        self.attention = Attention(embed_size)
        self.grader = Grader(embed_size)

    def forward(self, x, x_len, seq_length):
        x = self.embedding(x) # shape (batch_size, seq_length) -> (batch_size, seq_length, embed_size)
        batch_size = x.size(0)

        #######################################################################
        # Encoder
        encoder_output, encoder_state = self.encoder(x, x_len)
        
        #######################################################################
        # Decoder
        decoder_input = self.embedding(torch.LongTensor(np.random.choice(self.vocab_size, (batch_size, 1))).to(device)) # (batch_size, 1, embed_size)
        context, attention = self.attention(decoder_input, encoder_output)
        decoder_input = torch.cat([decoder_input, context], dim=2).to(device)
        decoder_state = encoder_state

        outputs = torch.zeros(seq_length, batch_size, 2).to(device)
        for t in range(seq_length):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            outputs[t] = self.grader(decoder_output.squeeze(1))

            context, attention = self.attention(decoder_output, encoder_output)
            decoder_input = torch.cat([decoder_output, context], dim=2).to(device)
            
        outputs = outputs.transpose(0, 1) # (batch_size, seq_length, 2)

        return outputs

class Model:
    def __init__(self, lang):
        embed_size = 256
        hidden_size = 512
        self.model = Seq2seq(lang.num_words, embed_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=2)

    def save_model(self, epoch):
        save_dir = './saved_model/'
        if not os.path.isdir(save_dir): os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_dir + 'attention_c_%d' % epoch)

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
                token_len = token_len.to(device)
                seq_len = label_input.size(1)

                outputs = self.model(token_input, token_len, seq_len).contiguous().view(-1, 2) # (batch_size * seq_length, 2)
                labels = nn.utils.rnn.pad_sequence(label_input, batch_first=True, padding_value=2).view(-1).to(device)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                if (batch_num+1) % 1000 == 0:
                    end_time=time.time()
                    print("Batch %d | Time %0.2fm" % (batch_num+1, (end_time-start_time)/60))
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
            for (data, instance_ids) in dataloader:
                x_len = torch.LongTensor([len(seq) for seq in data]).to(device)
                x = nn.utils.rnn.pad_sequence(data, batch_first=True).to(device)

                outputs = F.softmax(self.model(x, x_len), dim=2)[:, :, 1] # (batch_size, seq_length, 2)
                for batch_num in range(outputs.size(0)):
                    seq_length = x_len[batch_num]
                    for i in range(seq_length):
                        instance_id = instance_ids[batch_num][i]
                        prediction = outputs[batch_num][i]
                        predictions[instance_id] = prediction.item()
        return predictions
