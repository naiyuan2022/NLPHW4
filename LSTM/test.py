from util import *
from scipy.sparse import csr_matrix
import pickle
from Model import nn_LSTM
from torch.optim.lr_scheduler import LambdaLR
import os

hidden_size = 256
seq_length = 25
root = '/Users/lizhiyuan/学习/深度学习与NLP/W9大作业/LSTM-chinese-novel-generration-baseline-master/source/'

with open(root + "X_train.pickle", 'rb') as handle:
    X_train = pickle.load(handle)
with open(root + "y_train.pickle", 'rb') as handle:
    y_train = pickle.load(handle)
with open(root + "chars.pickle", 'rb') as handle:
    chars = pickle.load(handle)
with open(root + "vocab_size.pickle", 'rb') as handle:
    vocab_size = pickle.load(handle)


rnn = nn_LSTM(vocab_size, hidden_size, vocab_size)
for batch in get_batch(X_train, y_train, seq_length):
    X_batch, y_batch = batch
rnn.load_state_dict(torch.load(root + 'save_model.pth'))
rnn.eval()
print(sample_chars(rnn, X_batch[0], rnn.initHidden_test(), chars, 2000))
