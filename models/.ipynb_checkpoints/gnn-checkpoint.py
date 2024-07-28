import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from . import gnn_iclr
import numpy
from fewshot_re_kit.plot_tsne.sava_x import save_x

class GNN(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, N, hidden_size=230):
        '''
        N: Num of classes
        '''
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.gnn_obj = gnn_iclr.GNN_nl(N, self.node_dim, nf=96, J=1)
        self.drop = nn.Dropout()

    def forward(self, support, query, N, K, Q, label, model, is_training):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        # save original vector
        # name = 'beginning'
        # if not save_x(support, label, name, N * Q):
        #     print('save x_beginning_data failure!')

        support,s_head,s_tail = self.sentence_encoder(support)
        query,q_head,q_tail = self.sentence_encoder(query)
        support = support.view(-1, N, K, self.hidden_size)
        query = query.view(-1, N * Q, self.hidden_size)

        # cnn for entity
        zero_s = Variable(torch.zeros((s_head.size(0), N)).cuda())
        zero_q = Variable(torch.zeros((q_head.size(0), N)).cuda())
        # zero_s = Variable(torch.zeros((s_head.size(0), N)))
        # zero_q = Variable(torch.zeros((q_head.size(0), N)))
        s_head = self.sentence_encoder.entity_cnn(s_head)
        s_head = torch.cat([s_head, zero_s], 1).view(-1, N, K, 230+N)
        # s_head = self.drop(s_head)
        s_tail = self.sentence_encoder.entity_cnn(s_tail)
        s_tail = torch.cat([s_tail, zero_s], 1).view(-1, N, K, 230+N)
        # s_tail = self.drop(s_tail)
        q_head = self.sentence_encoder.entity_cnn(q_head)
        q_head = torch.cat([q_head, zero_q], 1).view(-1, N*Q, 230+N)
        # q_head = self.drop(q_head)
        q_tail = self.sentence_encoder.entity_cnn(q_tail)
        q_tail = torch.cat([q_tail, zero_q], 1).view(-1, N*Q, 230+N)
        # q_tail = self.drop(q_tail)

        # initialize for entity
        # zero_s = Variable(torch.zeros((50,185)).cuda())
        # zero_q = Variable(torch.zeros((10, 185)).cuda())
        # s_head = torch.mean(s_head,1)
        # s_head = torch.cat([s_head,zero_s ], 1).view(-1, N, K, 235)
        # s_tail = torch.mean(s_tail,1)
        # s_tail = torch.cat([s_tail, zero_s], 1).view(-1, N, K, 235)
        # q_head = torch.mean(q_head,1)
        # q_head = torch.cat([q_head, zero_q], 1).view(-1, N*Q, 235)
        # q_tail = torch.mean(q_tail,1)
        # q_tail = torch.cat([q_tail, zero_q], 1).view(-1, N*Q, 235)

        B = support.size(0)
        NQ = query.size(1)
        D = self.hidden_size

        s_head = s_head.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230+N)
        s_tail = s_tail.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230+N)

        q_head = q_head.view(-1, 1, 230+N)
        q_tail = q_tail.view(-1, 1, 230+N)

        nodes_head = torch.cat([q_head,s_head], 1)
        nodes_tail = torch.cat([q_tail,s_tail], 1)



        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, D) # (B * NQ, N * K, D)
        query = query.view(-1, 1, D) # (B * NQ, 1, D)
        labels = torch.zeros((B * NQ, 1 + N * K, N)).cuda()
        # labels = torch.zeros((B * NQ, 1 + N * K, N))
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels[b][1 + i * K + k][i] = 1
        labels = Variable(labels)
        nodes = torch.cat([torch.cat([query, support], 1), labels], -1) # (B * NQ, 1 + N * K, D + N)

        #select nodes
        nodes=torch.cat([nodes,nodes_head,nodes_tail],1)
        # logits = self.gnn_obj(nodes) # (B * NQ, N)
        logits = self.gnn_obj(nodes, label, NQ)  #
        _, pred = torch.max(logits, 1)
        return logits, pred 
