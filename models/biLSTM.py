# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class BiLSTM(nn.Module):
    
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        
        self.batch_size = batch_size
        self.output_size = output_size
        #self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
        #self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        #self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_length, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear1_output=64
        self.linear = nn.Linear(self.hidden_size*4 , self.linear1_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(self.linear1_output, output_size)


    def forward(self, x):
        h_embedding = self.word_embeddings(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out