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


class GNN_adv(fewshot_re_kit.framework.FewShotREModel):

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
        N: Num of classes eg.5
        K: Num of instances for each class in the support set eg.5
        Q: Num of instances for each class in the query set eg.1
        '''
        # save original vector
        # name = 'beginning'
        # if not save_x(support, label, name, N * Q):
        #     print('save x_beginning_data failure!')

        #lable[[2, 4, 1, 3, 0],[1, 4, 3, 2, 0]]
        support, s_head, s_tail = self.sentence_encoder.word_embedding(support) #50*40*60  50*20*50
        query, q_head, q_tail = self.sentence_encoder.word_embedding(query) #10*40*60  10*20*50

#sentence
        s = Variable(support.data, requires_grad=True)
        q = Variable(query.data, requires_grad=True)

        support = self.sentence_encoder.cnn(s)#50*230
        query = self.sentence_encoder.cnn(q) #10*230
        #print(query.size())


        support = support.view(-1, N, K, self.hidden_size)#2*5*5*230
        query = query.view(-1, N * Q, self.hidden_size)#2*5*230
#entity
        s_h = Variable(s_head.data, requires_grad=True)
        s_t = Variable(s_tail.data, requires_grad=True)
        q_h = Variable(q_head.data, requires_grad=True)
        q_t = Variable(q_tail.data, requires_grad=True)



        # cnn for entity
        zero_s = Variable(torch.zeros((s_head.size(0), N)).cuda())
        zero_q = Variable(torch.zeros((q_head.size(0), N)).cuda())
        # zero_s = Variable(torch.zeros((s_head.size(0), N)))
        # zero_q = Variable(torch.zeros((q_head.size(0), N)))
        s_head = self.sentence_encoder.entity_cnn(s_h) #50*230

        s_head = torch.cat([s_head, zero_s], 1).view(-1, N, K, 230 + N) #2*5*5*235
        # s_head = self.drop(s_head)
        s_tail = self.sentence_encoder.entity_cnn(s_t) #50*230

        s_tail = torch.cat([s_tail, zero_s], 1).view(-1, N, K, 230 + N)#2*5*5*235
        # s_tail = self.drop(s_tail)
        q_head = self.sentence_encoder.entity_cnn(q_h)#10*230

        q_head = torch.cat([q_head, zero_q], 1).view(-1, N * Q, 230 + N)#2*5*235
        # q_head = self.drop(q_head)
        q_tail = self.sentence_encoder.entity_cnn(q_t)
        q_tail = torch.cat([q_tail, zero_q], 1).view(-1, N * Q, 230 + N)#2*5*235
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

        B = support.size(0)#2

        NQ = query.size(1)#5

        D = self.hidden_size#230

        s_head = s_head.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230 + N)
        s_tail = s_tail.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230 + N)


        q_head = q_head.view(-1, 1, 230 + N)
        q_tail = q_tail.view(-1, 1, 230 + N)

        #print(s_head.size())
        #print(q_head.size())
        nodes_head = torch.cat([q_head, s_head], 1)#10*26*235
        #print(nodes_head.size())
        #input(x)
        nodes_tail = torch.cat([q_tail, s_tail], 1)#10*26*235
        # print(support.shape)
        # print(support[0])
        # print(support[1])
        # print(support[0].shape)

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, D)  # (B * NQ, N * K, D) 10*25*230
        #support[0] * 5 + support[1] * 5

        # print(support.shape)
        # print(support[0])
        # print(support[1])
        # print(support[2])
        # print(support[3])
        # print(support[4])
        # print(support[5])
        # print(support[6])
        # print(support[7])
        # print(support[8])
        # print(support[9])
        # input(x)

        query = query.view(-1, 1, D)  # (B * NQ, 1, D) #10*1*230

        labels = torch.zeros((B * NQ, 1 + N * K, N)).cuda() #10*26*5
        # labels = torch.zeros((B * NQ, 1 + N * K, N))
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels[b][1 + i * K + k][i] = 1

        labels = Variable(labels) #10*26*5
        # print(labels.size())
        # input(x)
        nodes = torch.cat([torch.cat([query, support], 1), labels], -1)  # (B * NQ, 1 + N * K, D + N)
        # print(nodes[0][0][-7:])
        # print(nodes[0][1][-7:])
        # print(nodes[1][0][-7:])
        # print(nodes[1][1][-7:])
        # print(nodes[7][0][-7:])
        # print(nodes[7][1][-7:])
        # print(nodes[8][0][-7:])
        # print(nodes[8][1][-7:])

        # input(x)
        # print(nodes, nodes_head, nodes_tail)
        # print(nodes[0][0].__len__())# 10*26*235
        # print(nodes_head[0][0].__len__())
        # print(nodes_tail[0][0].__len__())
        # input(x)
        nodes = torch.cat([nodes, nodes_head, nodes_tail], 1 )#10*78*235 sentences have label but entity not



        # print(nodes.shape)
        # input(x)
        # logits = self.gnn_obj(nodes) # (B * NQ, N)
        logits = self.gnn_obj(nodes, label, NQ)  #10 * 5

        _, pred = torch.max(logits, 1)


# Add Adversarial
        if is_training:
            loss = model.loss(logits, label)
            loss.backward()

#sentence_adv
            s_grad = s.grad
            norm_s = torch.norm(s_grad)
            perd_s = 0.1 * s_grad / norm_s
            support = self.sentence_encoder.cnn(Variable(s.data + perd_s.data))

            q_grad = q.grad
            norm_q = torch.norm(q_grad)
            perd_q = 0.1 * q_grad / norm_q
            query = self.sentence_encoder.cnn(Variable(q.data + perd_q.data))

            support = support.view(-1, N, K, self.hidden_size)
            query = query.view(-1, N * Q, self.hidden_size)

            support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K,D)  # (B * NQ, N * K, D)
            query = query.view(-1, 1, D)  # (B * NQ, 1, D)
            labels = torch.zeros((B * NQ, 1 + N * K, N)).cuda()
            # labels = torch.zeros((B * NQ, 1 + N * K, N))
            for b in range(B * NQ):
                for i in range(N):
                    for k in range(K):
                        labels[b][1 + i * K + k][i] = 1
            labels = Variable(labels)
            nodes = torch.cat([torch.cat([query, support], 1), labels], -1)  # (B * NQ, 1 + N * K, D + N)


#entity_adv
            s_h_grad = s_h.grad
            norm_s_h = torch.norm(s_h_grad)
            perd_s_h = 0.1 * s_h_grad / norm_s_h
            s_head = Variable(s_h.data + perd_s_h.data)

            s_t_grad = s_t.grad
            norm_s_t = torch.norm(s_t_grad)
            perd_s_t = 0.1 * s_t_grad / norm_s_t
            s_tail = Variable(s_t.data + perd_s_t.data)

            q_h_grad = q_h.grad
            norm_q_h = torch.norm(q_h_grad)
            perd_q_h = 0.1 * q_h_grad / norm_q_h
            q_head = Variable(q_h.data + perd_q_h.data)

            q_t_grad = q_t.grad
            norm_q_t = torch.norm(q_t_grad)
            perd_q_t = 0.1 * q_t_grad / norm_q_t
            q_tail = Variable(q_t.data + perd_q_t.data)

            s_head = self.sentence_encoder.entity_cnn(s_head)
            s_head = torch.cat([s_head, zero_s], 1).view(-1, N, K, 230 + N)
            # s_head = self.drop(s_head)
            s_tail = self.sentence_encoder.entity_cnn(s_tail)
            s_tail = torch.cat([s_tail, zero_s], 1).view(-1, N, K, 230 + N)
            # s_tail = self.drop(s_tail)
            q_head = self.sentence_encoder.entity_cnn(q_head)
            q_head = torch.cat([q_head, zero_q], 1).view(-1, N * Q, 230 + N)
            # q_head = self.drop(q_head)
            q_tail = self.sentence_encoder.entity_cnn(q_tail)
            q_tail = torch.cat([q_tail, zero_q], 1).view(-1, N * Q, 230 + N)
            # q_tail = self.drop(q_tail)

            s_head = s_head.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230 + N)
            s_tail = s_tail.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230 + N)

            q_head = q_head.view(-1, 1, 230 + N)
            q_tail = q_tail.view(-1, 1, 230 + N)

            nodes_head = torch.cat([q_head, s_head], 1)
            nodes_tail = torch.cat([q_tail, s_tail], 1)

            nodes = torch.cat([nodes, nodes_head, nodes_tail], 1)
            # logits = self.gnn_obj(nodes) # (B * NQ, N)
            logits = self.gnn_obj(nodes, label, NQ)  #

            _, pred = torch.max(logits, 1)

        return logits, pred
