import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim

class Encoder(nn.Module):
    def __init__(self, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_dim = word_embedding_dim + pos_embedding_dim * 2
        self.conv = nn.Conv1d(self.embedding_dim, self.hidden_size, 3, padding=1)
        # self.conv = nn.Conv2d(in_channels=3, out_channels=230, kernel_size=(3,50),groups=1,padding=(1,0))
        self.entity_conv = nn.Conv1d(word_embedding_dim, self.hidden_size, 3, padding=1)
        self.pool = nn.MaxPool1d(max_length)
        self.entity_pool = nn.MaxPool1d(20)
        self.drop = nn.Dropout()

        # For PCNN
        self.mask_embedding = nn.Embedding(4, 3)
        self.mask_embedding.weight.data.copy_(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.mask_embedding.weight.requires_grad = False
        self._minus = -100

        # LSTM

        # self.bilstm = nn.LSTM(60, 230//2, num_layers=1, dropout=0.5, bidirectional=True, bias = False,batch_first=True)


    def forward(self, inputs):
        # return self.LSTM(inputs)
        return self.cnn(inputs)

    def cnn(self, inputs):
        x = self.conv(inputs.transpose(1,2))
        x = F.relu(x)
        x = self.pool(x)
        return x.squeeze(2) # n x hidden_size

    # def cnn(self, inputs):
    #     x = self.conv(inputs)
    #     x = F.relu(x)
    #     x = self.pool(x.squeeze(3))
    #     return x.squeeze(2)

    # def pcnn(self, inputs, mask):
    #     x = self.conv(inputs.transpose(1, 2))
    #     # x = self.conv(inputs) # n x hidden x length
    #     # x = x.squeeze(3)
    #     mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
    #     pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
    #     pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
    #     pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
    #     x = torch.cat([pool1, pool2, pool3], 1)
    #     x = x.squeeze(2) # n x (hidden_size * 3)
    #     return x

    def pcnn(self, inputs, mask):
        x = self.conv(inputs) # n x hidden x length
        x = x.squeeze(3)
        mask = 1 - self.mask_embedding(mask).transpose(1, 2) # n x 3 x length
        pool1 = self.pool(F.relu(x + self._minus * mask[:, 0:1, :]))
        pool2 = self.pool(F.relu(x + self._minus * mask[:, 1:2, :]))
        pool3 = self.pool(F.relu(x + self._minus * mask[:, 2:3, :]))
        x = torch.cat([pool1, pool2, pool3], 1)
        x = x.squeeze(2) # n x (hidden_size * 3)
        return x

    # def LSTM (self,inputs):
    #     #     # x = inputs.view(40,400,60)
    #     #     h0 = c0 = Variable(torch.zeros(2,inputs.size(0),230//2), requires_grad=False)
    #     #     h0=h0.cuda()
    #     #     c0=c0.cuda()
    #     #     bilstm_out,_ =self.bilstm(inputs,(h0, c0))
    #     #     # bilstm_out, _ = nn.utils.rnn.pad_packed_sequence(bilstm_out, batch_first=True)
    #     #     # bilstm_out = torch.transpose(bilstm_out, 0, 1)
    #     #     # bilstm_out = torch.transpose(bilstm_out,1,2)
    #     #     bilstm_out = F.tanh(bilstm_out)
    #     #     bilstm_out = self.drop(bilstm_out)
    #     #
    #     #     #LSTM-+-CNN
    #     #     out = self.cnn(bilstm_out)
    #     #     #LSTM
    #     #     # bilstm_out =torch.transpose(bilstm_out,1,2)
    #     #     # out = F.max_pool1d(bilstm_out,bilstm_out.size(2)).squeeze(2)
    #     #     return out

    def entity_cnn(self, inputs):
        x = self.entity_conv(inputs.transpose(1,2))
        x = F.relu(x)
        x = self.entity_pool(x)
        return x.squeeze(2) # n x
