import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

class Embedding(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5):
        nn.Module.__init__(self)

        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding
        unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        blk = torch.zeros(1, word_embedding_dim)
        # print(word_vec_mat)
        word_vec_mat = torch.from_numpy(word_vec_mat)
        self.word_embedding = nn.Embedding(word_vec_mat.shape[0] + 2, self.word_embedding_dim, padding_idx=word_vec_mat.shape[0] + 1)
        self.word_embedding.weight.data.copy_(torch.cat((word_vec_mat, unk, blk), 0))
        # print(self.word_embedding.weight.data)
        # Position Embedding
        self.pos1_embedding = nn.Embedding(80, pos_embedding_dim, padding_idx=0)
        # self.pos1_embedding.weight.data.copy_(torch.nn.init.orthogonal(torch.rand(80, 50), gain=1))
        self.pos2_embedding = nn.Embedding(80, pos_embedding_dim, padding_idx=0)
        # self.pos2_embedding.weight.data.copy_(torch.nn.init.orthogonal(torch.rand(80, 50), gain=1))
    def forward(self, inputs):

        word = inputs['word']
        pos1 = inputs['pos1']
        pos2 = inputs['pos2']
        mask = inputs['mask']
        mask0 = inputs['mask0']
        head = inputs['head']
        tail = inputs['tail']

        pos1 = mask0*pos1
        pos2 = mask0*pos2
        # print(self.word_embedding.weight)
        # x = torch.stack([self.word_embedding(word),
        #                     self.pos1_embedding(pos1)*1,
        #                     self.pos2_embedding (pos2)*1], 1)
        x = torch.cat([self.word_embedding(word),
                         self.pos1_embedding(pos1),
                         self.pos2_embedding(pos2)], 2)
        w = self.word_embedding(word)
        p1 =  self.word_embedding(pos1)
        p2 = self.pos1_embedding(pos2)
        head =self.word_embedding(head)
        tail =self.word_embedding(tail)
        return x,head,tail
        # return x


