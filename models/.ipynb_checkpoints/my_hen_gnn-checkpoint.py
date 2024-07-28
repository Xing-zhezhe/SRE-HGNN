import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import Data
from . import my_hen_layers
import scipy.sparse as sp

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor



class hen_GNN(nn.Module):
    def __init__(self, N, input_features, nf, J): #5,235,96,1
        super(hen_GNN, self).__init__()
        # self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J
        self.num_layers = 3
        self.GCN_layer_config = [512, 5]
        self.GCN_dropout = 0.5
        self.item_feat = []
        self.GCN = self.build_model()

    def build_graphs(self, features, adjs):
        pyg_graphs = []
        for feat, adj in zip(features, adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)

        return pyg_graphs

    def forward(self, adj, sen_adj, entity_adj, total_graph, sen_graph, entity_graph, x, lable, NQ, is_training):


        #label :[[1, 2, 4, 0, 3],[1, 3, 2, 4, 0]] 2*5
        #NQ : 5

        GCN_out = []
        # print(type(x[0][0]))
        # print(x)
        # print(adj[0])
        # input(x)
        for i in range(0, len(adj)):
            GCN_out.append(self.GCN([x[i], adj[i]]))
        GCN_outputs = GCN_out #10*78*5

        GCN_outputs = torch.stack(GCN_outputs)

        return GCN_outputs[:, 0, :] #10*5

    def build_model(self):
        input_dim =  self.input_features
        # GCN Layer
        GCN_layers = nn.Sequential()

        layer = my_hen_layers.GCNLayer(input_dim=input_dim,
                         hidden_dim_1 =self.GCN_layer_config[0],
                         
                         output_dim=self.GCN_layer_config[-1],
                         GCN_dropout=self.GCN_dropout)
        GCN_layers.add_module(name="GCN_layer_{}".format(0), module=layer)

        return GCN_layers