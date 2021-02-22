# _*_ coding: utf-8 _*_
#https://github.com/prakashpandey9/Text-Classification-Pytorch
import os
import sys
import torch
from torch.nn import functional as F
import numpy as np
import torchtext
import spacy
from torchtext import data
from torchtext import datasets
from torchtext import vocab
from torchtext.vocab import Vectors, GloVe

# tokenizer function using spacy
'''
en = spacy.load('en')
def tokenizer(s): 
    return [tok.text for tok in en.tokenizer(s)]
'''
def load_dataset(test_sen=None):

    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    
    """
    
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField()#tensor_type=torch.FloatTensor)
    #train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)



    #txt_field = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)
    #label_field = data.LabelField()#

    #data_fields = [
    #    ('text', txt_field), 
    #    ('class', label_field), 
    #]

    #train_data, valid_data, test_data = data.TabularDataset.splits(path='/home/vasco/Documents/b-t4sa/', 
    #                                            format='csv', 
    #                                            train='clean_train.csv',
    #                                            validation='clean_val.csv',
    #                                            test='clean_test.csv',
    #                                            fields=data_fields, 
    #                                            skip_header=True)

    vects = vocab.Vectors('glove.twitter.27B.200d.txt', '/home/vasco/Documents/b-t4sa/gloveembeddings/')

    # Read data

    train_data, val_data, test_data = data.TabularDataset.splits(
        path='/home/vasco/Documents/b-t4sa/', format='csv', skip_header=True,
        train='clean_2_train.csv', validation='clean_2_val.csv', test='clean_2_test.csv',
        fields=[
            ('text', TEXT),
            ('label', LABEL)
        ])

    #test_data = data.TabularDataset.splits(
    #    path='/home/vasco/Documents/b-t4sa/clean_test.csv', format='csv', skip_header=True,
    #    fields=[
    #        ('text', TEXT),
    #        ('label', LABEL)
    #    ])

    TEXT.build_vocab(
        val_data, test_data,
        vectors=vects,
        max_size=205045
    )
    TEXT.build_vocab(
        train_data,
        vectors=vects,
        max_size=205045
    )
    
    LABEL.build_vocab(train_data, val_data, test_data)

    #TEXT.build_vocab(train_data, vectors=vects) # train_data, val_data, vectors=vects
    #LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    #print ("Label Length: " + str(len(LABEL.vocab)))

    #train_data, valid_data = train_data.split() # Further splitting of training_data to create new training_data & validation_data
    batch_sizes = 32
    #train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, val_data, test_data),batch_size=batch_sizes, sort=False, repeat=False, shuffle=(False, False, False)) #sort_key=lambda x: len(x.text)
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, val_data, test_data),batch_size=batch_sizes, sort=False, shuffle=False) #sort_key=lambda x: len(x.text)


    #for x in train_iter:
    #    print(f'{x}')
    #    exit()
    

    #For fusion
    #train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, val_data, test_data), batch_size=batch_sizes, sort=False, shuffle=False)#, shuffle=True)

    vocab_size = len(TEXT.vocab)
    #print (vocab_size)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter, batch_sizes
