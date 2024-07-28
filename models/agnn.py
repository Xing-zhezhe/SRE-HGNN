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

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)  # (bs, J, N, N) split: divide the tensor according 3 dim, 1 for each unit
    W = torch.cat(W, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x)  # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features) 10*78*(2*235)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J * nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input)  # out has size (bs, N, num_inputs) 10*78*470
        # if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x)  # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(x_size[0], x_size[1], self.num_outputs)
        return W, x


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], num_operators=1,
                 drop=False):
        super(Wcompute, self).__init__()
        self.num_features = nf
        self.operator = operator
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        # 二层卷积 Conv2d(in_channels,out_channels,kenel_size,stride) kenel_size=1 as well as shengwei or jiangwei
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        # **卷积层之后总会添加BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf * ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf * ratio[2])
        self.conv2d_4 = nn.Conv2d(nf * ratio[2], nf * ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf * ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        W1 = x.unsqueeze(2)  # 10*78*1*235

        W2 = torch.transpose(W1, 1, 2)  # size: bs x N x N x num_features   10*1*78*235

        W_new = torch.abs(W1 - W2)  # size: bs x N x N x num_features  10*78*78*235
       
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x num_features x N x N  10*235*78*78
        
        W_new = self.conv2d_1(W_new)  # 10*128*78*78

        W_new = self.bn_1(W_new)  # 10*128*78*78
        W_new = F.leaky_relu(W_new)  # jihuo
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)  # 10*128*78*78
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)  # 10*64*78*78
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)  # 10*64*78*78
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)  # 10*1*78*78
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x N x N x 1  10*78*78*1

        if self.activation == 'softmax':

            # W_new = W_new - W_id.expand_as(W_new) * 1e8
            # W_new = torch.transpose(W_new, 2, 3)
            #
            # # Applying Softmax
            # W_new = W_new.contiguous()
            # W_new_size = W_new.size()
            # W_new = W_new.view(-1, W_new.size(3))
            # W_new = F.softmax(W_new, dim=1)
            # W_new = W_new.view(W_new_size)
            # # Softmax applied
            # W_new = torch.transpose(W_new, 2, 3)

            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)
            # Applying Softmax
            W_new = W_new.contiguous()

            ### old version
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))  # size: (bs x N) x N

            p = 0.3 # 控制矩阵的稀疏程度
            mvalue = int(x.size(1) * p)
            [kval, _] = torch.kthvalue(W_new, mvalue, dim=1)  # torch.kthvalue(x, k, n) 沿着n维度返回第k小的数据
            W_index_drop = torch.ge(W_new, kval.unsqueeze(1).expand_as(W_new),
                                    out=None)  # 逐元素比较input和other，即是否 ( input >= other )
            W_index_drop = W_index_drop.cuda(device=W_new.device)
            W_new = W_new.mul(W_index_drop.type(torch.float32))  # 对应元素相乘
            W_new = W_new + torch.mul(torch.ones_like(W_index_drop.type(torch.float32), device=W_new.device),
                                      -1e3)  # 为0的元素赋值为-1000
            W_new = W_new - torch.mul(W_index_drop.type(torch.float32), -1e3)
            W_new = F.softmax(W_new, dim=1)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)

        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise (NotImplementedError)

        return W_new  # 10*78*78*2


class GNN_nl(nn.Module):
    def __init__(self, N, input_features, nf):  # 5,235,96,1
        super(GNN_nl, self).__init__()
        # self.args = args
        self.input_features = input_features
        self.nf = nf
        self.num_layers = 5
        
        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, N, 2, bn_bool=False)

    def forward(self, x, it, eval_iter, is_training, label, NQ):
        # label :[[1, 2, 4, 0, 3],[1, 3, 2, 4, 0]] 2*5
        # NQ : 5

        # heterogeneous graph : sentence + head + tail

        # x shape:10*78*235
        
        # save vector
        if(is_training==False and eval_iter==3000):
            if(it==1):
                name = 'After CNN'
                if not save_x(x, label, name, NQ):
                    print('save x_before_data failure!')
                    
        W_init = Variable(torch.eye(x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))  # 单位阵为邻接矩阵，每个类别的邻接矩阵都是单位矩阵
        for i in range(self.num_layers):
        
        ##################################保存每一层的输出之后的向量########################################
            
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])              
            x = torch.cat([x, x_new], 2)
            
            if(is_training==False and eval_iter==3000):
                if(it==1):
                    name = 'GNN_'+str(i+1)
                    if not save_x(x_new, label, name, NQ):
                        print('save GNN x failure!'+str(i+1)) 
            

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1] #10 *78*5
        
        if(is_training==False and eval_iter==3000):
                if(it==1):
                    name = 'end'
                    if not save_x(out, label, name, NQ):
                        print('save end x failure!') 
                        
        # return out[:, 25:50, :]  # , w_learn
        #return out[:, 0, :]  # 10*5
        return out

class GNN(fewshot_re_kit.xzz_framework.FewShotREModel):

    def __init__(self,sentence_encoder, N, hidden_size=230, alpha=0):
        '''
        N: Num of classes
        '''
        fewshot_re_kit.xzz_framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.node_dim = hidden_size + N
        self.alpha = alpha
        self.gnn_model = GNN_nl(N, self.node_dim, nf=64)
        self.fusion = nn.Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
        self.drop = nn.Dropout()
        self.temp = 10

    def forward(self, support, query, B, N, K, Q, label,
                model, it, eval_iter, is_training):
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

    #node feature
        
        support, s_head, s_tail = self.sentence_encoder.word_embedding(support)  # 25*40*60  25*20*50
        query, q_head, q_tail = self.sentence_encoder.word_embedding(query)  # 5*40*60  5*20*50
        
        if(is_training==False and eval_iter==3000):
        # save original vector
            if(it==1):
                name = 'beginning'
                if not save_x(support, label, name, N * Q):
                    print('save x_beginning_data failure!')
        
        # sentence
        s = Variable(support.data, requires_grad=True)
        q = Variable(query.data, requires_grad=True)

        support = self.sentence_encoder.cnn(s)  # 25*230
        query = self.sentence_encoder.cnn(q)  # 5*230

        support = support.view(-1, N, K, self.hidden_size) #1*5*5*230
        query = query.view(-1, N * Q, self.hidden_size)#1*5*230

        s_head = self.sentence_encoder.entity_cnn(s_head)# 1*5*5*230
        s_head = s_head.view(-1, N, K, 230)  # 2*5*5*235
        
        s_tail = self.sentence_encoder.entity_cnn(s_tail)# 1*5*5*230
        s_tail = s_tail.view(-1, N, K, 230)  # 2*5*5*235
        
        q_head = self.sentence_encoder.entity_cnn(q_head)# 1*5*230
        q_head = q_head.view(-1, N * Q, 230)  # 2*5*235
        
        q_tail = self.sentence_encoder.entity_cnn(q_tail)# 1*5*230
        q_tail = q_tail.view(-1, N * Q, 230)  # 2*5*235

        B = support.size(0)
        NQ = query.size(1)
        D = self.hidden_size
        #print(NQ)
        #print(s_head.shape)
        #input(x)

        s_head = s_head.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230)#5*25*230
        s_tail = s_tail.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, 230)#5*25*230

        q_head = q_head.view(-1, 1, 230)#5*1*230
        q_tail = q_tail.view(-1, 1, 230)#5*1*230

        nodes_head = torch.cat([q_head, s_head], 1)#5*26*230
        nodes_tail = torch.cat([q_tail, s_tail], 1)#5*26*230

        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1).contiguous().view(-1, N * K, D)  # (B * NQ, N * K, D) 5*25*230
        query = query.view(-1, 1, D)  # (B * NQ, 1, D) 5*1*230

    #node label
        #support sentence label 对应类别设置为1    query sentence label 全部类别设置为平均值
        labels_sen = torch.zeros((B * NQ, 1 + N * K, N)).cuda() #5*26*5

        #support label default 1
        for b in range(B * NQ):
            for i in range(N):
                for k in range(K):
                    labels_sen[b][1 + i * K + k][i] = 1
        #query label default avg
        for b in range(B * NQ):
            for k in range(K):
                labels_sen[b][0][k] = 1.0/K

        #entity label 全部填充为0
        zero_s = torch.zeros((B * NQ, 1 + N * K, N)).cuda()# the same shape to sentence label
        zero_q = torch.zeros((B * NQ, 1 + N * K, N)).cuda()
        labels_enti = torch.cat([zero_s, zero_q], 1)

        #total labels
        labels = torch.cat([labels_sen, labels_enti], 1)
       
        
        #print(query)
        #print(support)
        
        #total features
        nodes = torch.cat([query, support], 1) # (B * NQ, 1 + N * K, D + N)
        #print(nodes)
        #input(x)
        
        nodes = torch.cat([nodes, nodes_head, nodes_tail], 1)# B * NQ, 3 * (1 + N * K), D

        # self-attention and fusion block
        x = F.normalize(nodes, p=2, dim=2, eps=1e-12) # p:第二范数归一化，dim:在2维度的归一化 class * nodenum *dim
        x_trans = torch.transpose(x, 1, 2)
        att = torch.bmm(x, x_trans)

        lab_t = torch.transpose(labels, 1, 2)
        att_l = torch.bmm(labels, lab_t)

        mask_c = torch.cat([att.unsqueeze(1), att_l.unsqueeze(1)], dim=1)
        new_mask = self.fusion(mask_c).squeeze(1)  # 一层全连接 w1*x+w2*y  结果为Cf

        new_fea = torch.bmm(new_mask, nodes)  # X(1)
        lab_new = torch.mul(torch.bmm(new_mask, labels), 1 - self.alpha) + torch.mul(labels, self.alpha)  # Y(1)
        xx = torch.cat([new_fea, lab_new], dim=2)

        out_fea = self.gnn_model(xx, it, eval_iter, is_training, label, NQ)
        logits = out_fea[:, 0, :]
        #print("logits")
        #print(logits)
        logits = out_fea[:, 0, :].squeeze(-1) 
        #print("logits 2")
        #print(logits)
        
        _, pred = torch.max(logits, 1)
        # logits = self.gnn_obj(nodes, label, NQ)  #
        # _, pred = torch.max(logits, 1)
        return logits, pred, out_fea
        
        
        
        