import torch
from torch import nn
import numpy as np

class CharCNN(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(CharCNN, self).__init__()
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, embedding_length)
        self.embeddings.weight = nn.Parameter(weights, requires_grad=False)
        
        # This stackoverflow thread explains how conv1d works
        # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
        conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_length, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6)/3)
        conv2 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6-18)/(3*3))
        conv3 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18)/(3*3))
        conv4 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18-18)/(3*3))
        conv5 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU()
        ) # (batch_size, num_channels, (seq_len-6-18-18-18-18)/(3*3))
        conv6 = nn.Sequential(
            nn.Conv1d(in_channels=self.config.num_channels, out_channels=self.config.num_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3)
        ) # (batch_size, num_channels, (seq_len-6-18-18-18-18-18)/(3*3*3))
        
        # Length of output after conv6        
        conv_output_size = self.config.num_channels * ((self.config.seq_len - 96) // 27)
        
        linear1 = nn.Sequential(
            nn.Linear(conv_output_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear2 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.linear_size),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_keep)
        )
        linear3 = nn.Sequential(
            nn.Linear(self.config.linear_size, self.config.output_size),
            nn.Softmax()
        )
        
        self.convolutional_layers = nn.Sequential(conv1,conv2,conv3,conv4,conv5,conv6)
        self.linear_layers = nn.Sequential(linear1, linear2, linear3)
    
    def forward(self, x):
        embedded_sent = self.embeddings(x).permute(1,2,0) # shape=(batch_size,embed_size,seq_len)
        conv_out = self.convolutional_layers(embedded_sent)
        conv_out = conv_out.view(conv_out.shape[0], -1)
        linear_output = self.linear_layers(conv_out)
        return linear_output