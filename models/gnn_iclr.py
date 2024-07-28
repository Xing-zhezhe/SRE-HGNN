#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from fewshot_re_kit.plot_tsne.sava_x import save_x

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor


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

        W_new = self.conv2d_1(W_new)  # 10*192*78*78

        W_new = self.bn_1(W_new)  # 10*192*78*78
        W_new = F.leaky_relu(W_new)  # jihuo
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)  # 10*192*78*78
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)  # 10*96*78*78
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)  # 10*96*78*78
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)  # 10*1*78*78
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x N x N x 1  10*78*78*1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)

            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
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




class GNN_nl_omniglot(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl_omniglot, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers):
            module_w = Wcompute(self.input_features + int(nf / 2) * i,
                                self.input_features + int(nf / 2) * i,
                                operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=False)
            module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers,
                                    self.input_features + int(self.nf / 2) * (self.num_layers - 1),
                                    operator='J2', activation='softmax', ratio=[2, 1.5, 1, 1], drop=True)
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2,
                                bn_bool=True)

    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


class GNN_active(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_active, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2
        for i in range(self.num_layers // 2):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)

            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.conv_active_1 = nn.Conv1d(self.input_features + int(nf / 2) * 1, self.input_features + int(nf / 2) * 1, 1)
        self.bn_active = nn.BatchNorm1d(self.input_features + int(nf / 2) * 1)
        self.conv_active_2 = nn.Conv1d(self.input_features + int(nf / 2) * 1, 1, 1)

        for i in range(int(self.num_layers / 2), self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2',
                                    activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2,
                                bn_bool=False)

    def active(self, x, oracles_yi, hidden_labels):
        x_active = torch.transpose(x, 1, 2)
        x_active = self.conv_active_1(x_active)
        x_active = F.leaky_relu(self.bn_active(x_active))
        x_active = self.conv_active_2(x_active)
        x_active = torch.transpose(x_active, 1, 2)

        x_active = x_active.squeeze(-1)
        x_active = x_active - (1 - hidden_labels) * 1e8
        x_active = F.softmax(x_active)
        x_active = x_active * hidden_labels

        if self.args.active_random == 1:
            # print('random active')
            x_active.data.fill_(1. / x_active.size(1))
            decision = torch.multinomial(x_active)
            x_active = x_active.detach()
        else:
            if self.training:
                decision = torch.multinomial(x_active)
            else:
                _, decision = torch.max(x_active, 1)
                decision = decision.unsqueeze(-1)

        decision = decision.detach()

        mapping = torch.FloatTensor(decision.size(0), x_active.size(1)).zero_()
        mapping = Variable(mapping)
        if self.args.cuda:
            mapping = mapping.cuda()
        mapping.scatter_(1, decision, 1)

        mapping_bp = (x_active * mapping).unsqueeze(-1)
        mapping_bp = mapping_bp.expand_as(oracles_yi)

        label2add = mapping_bp * oracles_yi  # bsxNodesxN_way
        padd = torch.zeros(x.size(0), x.size(1), x.size(2) - label2add.size(2))
        padd = Variable(padd).detach()
        if self.args.cuda:
            padd = padd.cuda()
        label2add = torch.cat([label2add, padd], 2)

        x = x + label2add
        return x

    def forward(self, x, oracles_yi, hidden_labels):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers // 2):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        x = self.active(x, oracles_yi, hidden_labels)

        for i in range(int(self.num_layers / 2), self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            x = torch.cat([x, x_new], 2)

        Wl = self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, 0, :]


class GNN_nl(nn.Module):
    def __init__(self, N, input_features, nf, J):  # 5,235,96,1
        super(GNN_nl, self).__init__()
        # self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute(self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute(self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = Gconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2',
                                    activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(self.input_features + int(self.nf / 2) * self.num_layers, N, 2, bn_bool=False)

    def forward(self, x, label, NQ, K, N):
        # label :[[1, 2, 4, 0, 3],[1, 3, 2, 4, 0]] 2*5
        # NQ : 5
        # W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        # W_init = W_init.cuda()

        # # save vector
        # name = 'After CNN'
        # if not save_x(x, label, name, NQ):
        #     print('save x_before_data failure!')

        # heterogeneous graph : sentence + head + tail

        # x shape:10*78*235
        h = int(x.size(1) / 3)
        W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        for i in range(3):
            for m in range(h):
                for n in range(3):
                    W_init[i * h + m][m + n * h] = 1 / 3
        # 句子、头实体、尾实体 三种节点拼接在一起，邻接矩阵内容是句子和头实体尾实体有边
        # W_init.size= 78*78
        # 添加同类之间的边
        # add_W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        # add_edge_weight = torch.rand(1, 1)
        #
        # for i in range(N):
        #     for m in range(i * K + 1, (i + 1) * K + 1):
        #         for n in range(i * K + 1, (i + 1) * K + 1):
        #             add_W_init[m][n] = add_edge_weight
        #
        # add_W_init = Variable(add_W_init)
        # W_init = W_init + add_W_init
        # 输出到文件
        #         with open('out1.txt', 'w') as f:
        #             for i in range(add_W_init.shape[0]):
        #                 for j in range(add_W_init.shape[1]):
        #                     print(add_W_init[i][j].item(), file=f)
        # heterogeneous graph : sentence + (head  or tail)
        # h = int(x.size(1) / 2)
        # W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        # for i in range(2):
        #     for m in range(h):
        #         for n in range(2):
        #             W_init[i * h + m][m + n * h] = 1 / 2
        # w init
        W_init = Variable(W_init)
        W_init = W_init.unsqueeze(-1).expand(x.size(0), -1, -1, -1).contiguous()  # 10*78*78*1
        W_init = W_init.cuda()
        # W_init = W_init
        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)#MLP(abs())
            # 10.78.78.2   10.78.78.2
            x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])  # ALW
            # 10.78.48   10.78.48
            x = torch.cat([x, x_new], 2)
            # 10.78.283  10.78.331
        # save
        # name = 'GNN-1'
        # if not save_x(x, label, name, NQ):
        #     print('save x_after_data failure!')

        Wl = self.w_comp_last(x, W_init)  # 10.78.78.2
        out = self.layer_last([Wl, x])[1]  # 10.78.5

        # save
        # name = 'end'
        # if not save_x(out, label, name, NQ):
        #     print('save x_end_data failure!')

        return out[:, 0, :]  # 10*5


class Wcompute_c(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1], num_operators=1,
                 drop=False):
        super(Wcompute_c, self).__init__()
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

        W_new = self.conv2d_1(W_new)  # 10*192*78*78

        W_new = self.bn_1(W_new)  # 10*192*78*78
        W_new = F.leaky_relu(W_new)  # jihuo
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)  # 10*192*78*78
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)  # 10*96*78*78
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)  # 10*96*78*78
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)  # 10*1*78*78
        W_new = torch.transpose(W_new, 1, 3)  # size: bs x N x N x 1  10*78*78*1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3)

            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
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


class GAT_nl(nn.Module):
    def __init__(self, N, input_features, nf, J):  # 5,235,96,1
        super(GAT_nl, self).__init__()
        # self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J
        self.nheads = 8
        self.alpha = 0.2
        self.dropout = 0.6
        self.nhid = 8
        self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                module_w = Wcompute_c(self.input_features, nf, operator='laplace', activation='softmax', ratio=[2, 2, 1, 1])
                module_l = GATconv(self.input_features, int(nf / 2), 2)
            else:
                module_w = Wcompute_c(self.input_features + int(nf / 2) * i, nf, operator='laplace', activation='softmax',
                                    ratio=[2, 2, 1, 1])
                module_l = GATconv(self.input_features + int(nf / 2) * i, int(nf / 2), 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute_c(self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='laplace',
                                    activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = GATconv(self.input_features + int(self.nf / 2) * self.num_layers, N, 2, bn_bool=False)

    def forward(self, x, label, NQ, K, N):
        # label :[[1, 2, 4, 0, 3],[1, 3, 2, 4, 0]] 2*5
        # NQ : 5
        # W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))
        # W_init = W_init.cuda()

        # # save vector
        # name = 'After CNN'
        # if not save_x(x, label, name, NQ):
        #     print('save x_before_data failure!')

        # heterogeneous graph : sentence + head + tail

        # x shape:10*78*235
        h = int(x.size(1) / 3)
        W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        for i in range(3):
            for m in range(h):
                for n in range(3):
                    W_init[i * h + m][m + n * h] = 1 / 3
        # 句子、头实体、尾实体 三种节点拼接在一起，邻接矩阵内容是句子和头实体尾实体有边
        # W_init.size= 78*78
        # 添加同类之间的边
        # add_W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        # add_edge_weight = torch.rand(1, 1)
        #
        # for i in range(N):
        #     for m in range(i * K + 1, (i + 1) * K + 1):
        #         for n in range(i * K + 1, (i + 1) * K + 1):
        #             add_W_init[m][n] = add_edge_weight
        #
        # add_W_init = Variable(add_W_init)
        # W_init = W_init + add_W_init
        # 输出到文件
        #         with open('out1.txt', 'w') as f:
        #             for i in range(add_W_init.shape[0]):
        #                 for j in range(add_W_init.shape[1]):
        #                     print(add_W_init[i][j].item(), file=f)
        # heterogeneous graph : sentence + (head  or tail)
        # h = int(x.size(1) / 2)
        # W_init = torch.zeros(x.size(1), x.size(1)).cuda()
        # for i in range(2):
        #     for m in range(h):
        #         for n in range(2):
        #             W_init[i * h + m][m + n * h] = 1 / 2
        # w init
        W_init = Variable(W_init)
        W_init = W_init.unsqueeze(-1).expand(x.size(0), -1, -1, -1).contiguous()  # 10*78*78*1
        W_init = W_init.cuda()
        W_init_size = W_init.size()
        # W_init = W_init

        x_j = []
        out = []

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)
            Wi = Wi.view(W_init_size)

            for j in range(x.size(0)):
                x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x[j]])[1])  # ALW
                # 10.78.48   10.78.48
                x_j.append(torch.cat([x[j], x_new], 2))
            x_j = torch.stack(x_j)
            x = x_j
        Wl = self.w_comp_last(x, W_init)  # 10.78.78.2

        for j in range(x.size(0)):
            Wl = Wl.view(W_init_size)
            out.append(self.layer_last([Wl, x[j]])[1])  # 10.78.5

        out = torch.stack(out)

        return out[:, 0, :]  # 10*5


class GATconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=True):
        super(GATconv, self).__init__()
        self.J = J
        self.num_inputs = nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.nheads = 8
        self.alpha = 0.2
        self.dropout = 0.6
        self.nhid = 512
        self.attentions = [
            GraphAttentionLayer(in_features=self.num_inputs, out_features=self.nhid, dropout=self.dropout,
                                alpha=self.alpha,
                                concat=True) for _ in
            range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(self.nhid * self.nheads, self.num_outputs, dropout=self.dropout,
                                           alpha=self.alpha,
                                           concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, input):
        W = input[0]
        x = input[1]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, W) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, W))  # 第二层的attention layer
        x = F.log_softmax(x, dim=1)
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], self.num_outputs)

        return W, x



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        device = torch.device('cuda')
        W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(W.data, gain=1.414)
        a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(a.data, gain=1.414)
        h = h.to(device)
        W = W.to(device)
        print(h.shape)
        print(W.shape)

        # input(xxxxxxxxxxxxxxx)

        Wh = torch.mm(h, W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)

        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        a_input = a_input.to(device)
        a = a.to(device)
        e = self.leakyrelu(torch.matmul(a_input, a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15 * torch.ones_like(e)

        attention = torch.where(adj > 0, e, zero_vec)  # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)  # elu-激活函数
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == '__main__':
    # test modules
    bs = 4
    nf = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())


