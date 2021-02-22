import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier_2(nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier_2, self).__init__()

        self.embedding_dim = embedding_length
        self.hidden_dim = hidden_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        

        self.embedding = nn.Embedding(vocab_size, embedding_length)
        self.embedding.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.

        self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=1)

        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
        autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))


    def forward(self, input_sentence, batch_size=None):
        
        if batch_size is None:
            h_0 = Variable(torch.randn(1, self.batch_size, self.hidden_dim).cuda()) # Initial hidden state of the LSTM
            c_0 = Variable(torch.randn(1, self.batch_size, self.hidden_dim).cuda()) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.randn(1, batch_size, self.hidden_dim).cuda())
            c_0 = Variable(torch.randn(1, batch_size, self.hidden_dim).cuda())

        input = self.embedding(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)

        outputs, (ht, ct) = self.lstm(input, (h_0, c_0))

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output