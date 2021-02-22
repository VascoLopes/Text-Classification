import torch
from torch import nn
import numpy as np

class fastText(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(fastText, self).__init__()
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        
        # Hidden Layer
        self.fc1 = nn.Linear(embedding_length, hidden_size)
        
        # Output Layer
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Softmax non-linearity
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        embedded_sent = self.embeddings(x)#.permute(1,0,2)
        embedded_sent = embedded_sent.transpose(1,2) # (batch_size, sequence_length, embed_size) -> (batch_size, embed_size, sequence_length)
        h = self.fc1(embedded_sent.mean(1))
        z = self.fc2(h)
        return self.softmax(z)