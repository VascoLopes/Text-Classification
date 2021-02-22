# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import f1_score


class TextCNN(nn.Module):
	# Based on Convolutional Neural Networks for Sentence Classification
	# https://arxiv.org/abs/1408.5882
    
	def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
		super(TextCNN, self).__init__()
		filter_sizes = [1,2,3,5]
		num_filters = 36
		self.embedding = nn.Embedding(vocab_size, embedding_length)
		self.embedding.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32), requires_grad=False)
		self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embedding_length)) for K in filter_sizes])
		self.dropout = nn.Dropout(0.1) # might be 0.2
		#fixed_length = 50
		#self.maxpools = [nn.MaxPool2d((fixed_length+1-i,1)) for i in filter_sizes] # might have maxpoll
		self.fc1 = nn.Linear(len(filter_sizes)*num_filters, output_size)

	def forward(self, x):
		x = self.embedding(x)  
		x = x.unsqueeze(1)  
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
		x = torch.cat(x, 1)
		x = self.dropout(x)  
		logit = self.fc1(x)  
		return logit