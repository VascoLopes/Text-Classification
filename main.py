import os
import pickle
import time
import load_data
import torch
import torch.nn.functional as F
from torch.nn import *
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
np.set_printoptions(threshold=np.inf)
import copy
import pandas as pd
from models.LSTM import LSTMClassifier
from models.LSTM_Attn import AttentionModel
from models.selfAttention import SelfAttention
from models.RNN import RNN
from models.RCNN import RCNN
from models.CNN import CNN
from models.biLSTM import BiLSTM
from models.TextCNN import TextCNN
from models.vdcnn import VDCNN
from models.QRNN import QRNN
from models.LSTMClassifier import LSTMClassifier_2
from models.fastText import fastText

# Read dataset and embeddings
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter, batch_size_loadData = load_data.load_dataset() 

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
# Train the model for n epochs
def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)
    #optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.3, momentum=0.9)
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not 32):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        
        loss = loss_fn(prediction, target)

        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        # Print every 500 batches to give feedback
        if steps % 500 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


# Evaluate model in a iterator dataset
def eval_model(model, val_iter, mode='val'):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not 32):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            
            # Store predictions and ground truth in lists
            if(mode == 'test'):
                for i in range(0, len(prediction.detach().cpu().numpy())):
                    predictions.append(prediction.detach().cpu().numpy()[i].argmax())
                    ground_truth.append(target.detach().cpu().numpy()[i])

            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)
	
# Evaluate model in a iterator dataset
def eval_model_fusion(model, val_iter, mode='val', filename="fusion.csv"):
    # Open file and write header
    f = open(filename, 'w')
    f.write('neg\tneu\tpos\tfileLayer\n')

    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for batch in val_iter:
            text = batch.text[0]
            #if (text.size()[0] is not 32):
            #    continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()

            #print(len(text))
            prediction, layerLinear = model(text, len(text))
            for i in range(0, len(prediction.detach().cpu().numpy())):
                #print(len(layerLinear.detach().cpu()[i].numpy()))
                #exit()
                # Get prediction values between 0 and 1
                a,b,c = F.softmax(prediction.detach().cpu()[i])
                # Get last one before last layer values
                x_str = np.array2string(layerLinear.detach().cpu()[i].numpy()).replace('\n', '')
                # Write them
                f.write(str(a.numpy())+"\t"+str(b.numpy())+"\t"+str(c.numpy())+"\t"+ x_str +"\n")
                f.flush()

            loss = loss_fn(prediction, target)

            # Store predictions and ground truth in lists
            '''
            if(mode == 'test'):
                for i in range(0, len(prediction.detach().cpu().numpy())):
                    predictions.append(prediction.detach().cpu().numpy()[i].argmax())
                    ground_truth.append(target.detach().cpu().numpy()[i])
            '''
            
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

            steps += 1
            if steps % 200 == 0:
                print (f'Idx: {steps+1}, Loss: {loss.item():.4f}, Accuracy: {acc.item(): .2f}%')

    
    f.close()
    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


# predict on new data
def predict_new_data(model, val_iter, mode='val', filename="fusion.csv"):
    # Open file and write header
    f = open(filename, 'w')
    f.write('neg\tneu\tpos\n')

    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    steps = 0
    with torch.no_grad():
        for batch in val_iter:
            text = batch.caption_text[0]
            #if (text.size()[0] is not 32):
            #    continue
            if torch.cuda.is_available():
                text = text.cuda()

            #print(len(text))
            prediction = model(text, len(text))
            for i in range(0, len(prediction.detach().cpu().numpy())):
                #print(len(layerLinear.detach().cpu()[i].numpy()))
                #exit()
                # Get prediction values between 0 and 1
                a,b,c = F.softmax(prediction.detach().cpu()[i])
                # Get last one before last layer values
                # Write them
                f.write(str(a.numpy())+"\t"+str(b.numpy())+"\t"+str(c.numpy())+"\n")
                f.flush()


            # Store predictions and ground truth in lists
            '''
            if(mode == 'test'):
                for i in range(0, len(prediction.detach().cpu().numpy())):
                    predictions.append(prediction.detach().cpu().numpy()[i].argmax())
                    ground_truth.append(target.detach().cpu().numpy()[i])
            '''

            steps += 1
            if steps % 200 == 0:
                print (f'Idx: {steps+1}')

    
    f.close()
    return 0,0


''' Hyper Parameters '''
learning_rate = 1e-3
batch_size = batch_size_loadData
output_size = 3 # 3 classes
hidden_size = 256
embedding_length = 200
epochs = 20

''' Different Networks Possibilities '''
#LSTM
#print("#### Using LSTM ####")
#modelUsed = "LSTM"
#model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#LSTMClassifier
#print("#### Using LSTMClassifier ####")
#modelUsed = "LSTMClassifier"
#model = LSTMClassifier_2(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#FastText
#print("#### Using FastText ####")
#modelUsed = "FastText"
#model = fastText(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#BI-LSTM
#print("#### Using BiDirectional LSTM ####")
#modelUsed = "BILSTM"
#model = BiLSTM(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#LSTM ATTN
#modelUsed = "LSTM_ATTN"
#print ("#### Using LSTM ATTN ####")
#model = AttentionModel(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#CNN
#modelUsed = "CNN"
#print ("#### Using CNN ####")
#model = CNN(batch_size, output_size, 1, 128, [3, 4, 5], 1, 0, 0.8, vocab_size, embedding_length, word_embeddings)

#RNN
#print ("#### Using RNN ####")
#modelUsed = "RNN"
#model = RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#RCNN
modelUsed = "RCNNFINAL"
print ("#### Using RCNN####")
model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#QRNN
#modelUsed = "QRNN"
#print ("#### Using QRNN ####")
#model = QRNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#TextCNN
#print ("#### Using Text CNN ####")
#modelUsed = "TextCNN"
#model = TextCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#VDCNN
#print("#### Using VDCNN ####")
#modelUsed = "VDCNN17"
#model = VDCNN(depth=17, vocab_size=vocab_size, embedding_length=embedding_length, weights = word_embeddings, n_classes=output_size, k=2)

#QRNN
#print("#### Using QRNN ####")
#modelUsed = "QRNN"
#model = QRNN(batch_size, output_size, 300, vocab_size, embedding_length, word_embeddings, num_layers=4, kernel_size=2, pooling='fo', zoneout=0.5, dropout=0.3, dense=True)


# Loss function
loss_fn = F.cross_entropy


# Store values
val_acc_history = []
train_acc_history = []
train_loss_history = []
val_loss_history = []
best_acc = 0.0 # the best validation accuracy so far
epoch_best_model = 0 # the epoch in which the best validation model was found

def train():
    global best_acc, epoch_best_model
    print("#### Start Training [...] ####")
    val_acc = 0
    for epoch in range(0, epochs):
        #Conduct train
        train_loss, train_acc = train_model(model, train_iter, epoch) #
        train_loss_history.append(train_loss)# store train loss
        train_acc_history.append(train_acc)  # store train accuracy

        #Conduct Validation
        val_loss, val_acc = eval_model(model, valid_iter)
        val_loss_history.append(val_loss)  # store val loss
        val_acc_history.append(val_acc)    # store val accuracy
        
        # Print final values for each epoch
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

        # If current model has the best val accuracy ever, store it:
        if val_acc > best_acc:
            epoch_best_model = epoch
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    return best_model_wts
'''
start_time = time.time()
# Train
best_model = train()

# load best model weights, since last epoch might not be the best
model.load_state_dict(best_model)

# Save best model for future use
modelUsed = modelUsed+"bestModel_epoch"+str(epoch_best_model+1)
torch.save(model.state_dict(), '/home/vasco/Documents/Text-Classification-Pytorch/bestNetworks/'+modelUsed)

# Test model
print("\n#### Start Testing [...] ####")
predictions = []
ground_truth = []
test_loss, test_acc = eval_model(model, test_iter, mode='test')
# Print results
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%\n')

with open(modelUsed+'_predictions', 'wb') as fp:
    pickle.dump(predictions, fp)
with open(modelUsed+'_groundtruth', 'wb') as fp:
    pickle.dump(ground_truth, fp)

# Print informations about network and train
print("#### History and parameters [...] ####")
print(f'Train Accuracy History: {train_acc_history}')
print(f'Train Loss History: {train_loss_history}')
print(f'Validation Accuracy History: {val_acc_history}')
print(f'Validation Loss History: {val_loss_history}')
print(f'Hyperparameters: LR:{learning_rate} | BatchSize:{batch_size} | Classes:{output_size} | HiddenSize:{hidden_size} | GloveDim:{embedding_length} | NrEpochs:{epochs}')
print(f'Time elapsed: {time.time() - start_time}')

'''

model = RCNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)

#'''
# For fusion
model.load_state_dict(torch.load("/home/vasco/Documents/Text-Classification-Pytorch/bestNetworks/RCNNFINALbestModel_epoch3"))
model.cuda()
#print(model2)
predictions = []
ground_truth = []
test_loss, test_acc = predict_new_data(model, train_iter, mode='test', filename="train_img_caption.csv")
print(f'Test Loss: {test_loss:.3f}, TestTrain Acc: {test_acc:.2f}%\n')
###Val
predictions = []
ground_truth = []
test_loss, test_acc = predict_new_data(model, valid_iter, mode='test', filename="val_img_caption.csv")
print(f'Test Loss: {test_loss:.3f}, TestVal Acc: {test_acc:.2f}%\n')

### Test
predictions = []
ground_truth = []
test_loss, test_acc = predict_new_data(model, test_iter, mode='test', filename="test_img_caption.csv")
print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%\n')

###
#'''



'''
### Just for fun
# Let us now predict the sentiment on a single sentence just for the testing purpose.
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
'''