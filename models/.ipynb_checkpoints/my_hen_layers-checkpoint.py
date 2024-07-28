import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter
from . import GCN_layer
import numpy as np

import copy


class GCNLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim_1,
                 
                 output_dim,
                 GCN_dropout):
        super(GCNLayer, self).__init__()
        self.dropout = GCN_dropout
        # self.lin = nn.Linear(input_dim,input_dim)
        self.gc1 = GCN_layer.GraphConvolution(input_dim, hidden_dim_1, self.dropout)
        # self.gc2 = GCN_layer.GraphConvolution(hidden_dim_1, hidden_dim_2, self.dropout)
        self.gc3 = GCN_layer.GraphConvolution(hidden_dim_1, output_dim, self.dropout)
        self.dropout = GCN_dropout

    def forward(self, group):
        # graph = copy.deepcopy(total_graph)
        feats = group[0]
        adj = group[1]

        # feats = graph.x
        x = F.relu(self.gc1(feats, adj))
        # x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        return F.softmax(x, dim=1)
