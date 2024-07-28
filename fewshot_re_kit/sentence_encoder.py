import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from . import network

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x,head,tail  = self.embedding(inputs)
        x = self.encoder(x)
        return x,head,tail

    def word_embedding(self,inputs):

        x = self.embedding(inputs)

        return x

    def cnn(self,inputs):
        # print("input")
        # print(inputs)
        x = self.encoder(inputs)
        # print("cnn")
        # print(x)
        return x

    def entity_cnn(self,inputs):
        x = self.encoder.entity_cnn(inputs)
        return x

class PCNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=50, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder.pcnn(x, inputs['mask'])
        return x

    def word_embedding(self,inputs):
        x = self.embedding(inputs)

        return x

    def pcnn(self,x,inputs):
        x = self.encoder.pcnn(x, inputs['mask'])
        return x
