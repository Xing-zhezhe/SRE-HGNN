import json
import os
import sys
import multiprocessing
import numpy as np
import random
import torch
from torch.autograd import Variable
from collections import defaultdict
import networkx as nx
import pickle
import numpy as np
import scipy.sparse as sparse


class FileDataLoader:
    def next_batch1(self, B, N, K, Q):
        '''
        B: batch size.
        N: the number of relations for each batch
        K: the number of support instances for each relation
        Q: the number of query instances for each relation
        return: support_set, query_set, query_label
        '''
        raise NotImplementedError


class JSONFileDataLoader(FileDataLoader):
    def _load_preprocessed_file(self):
        name_prefix = '.'.join(self.file_name.split('/')[-1].split('.')[:-1])
        word_vec_name_prefix = '.'.join(self.word_vec_file_name.split('/')[-1].split('.')[:-1])
        processed_data_dir = '_processed_data'
        if not os.path.isdir(processed_data_dir):
            return False
        word_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_word.npy')
        pos1_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos1.npy')
        pos2_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_pos2.npy')
        mask_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask.npy')
        mask0_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_mask0.npy')
        head_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_head.npy')
        tail_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_tail.npy')
        length_npy_file_name = os.path.join(processed_data_dir, name_prefix + '_length.npy')
        rel2scope_file_name = os.path.join(processed_data_dir, name_prefix + '_rel2scope.json')
        word_vec_mat_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy')
        word2id_file_name = os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json')
        if not os.path.exists(word_npy_file_name) or \
                not os.path.exists(pos1_npy_file_name) or \
                not os.path.exists(pos2_npy_file_name) or \
                not os.path.exists(mask_npy_file_name) or \
                not os.path.exists(mask0_npy_file_name) or \
                not os.path.exists(length_npy_file_name) or \
                not os.path.exists(rel2scope_file_name) or \
                not os.path.exists(word_vec_mat_file_name) or \
                not os.path.exists(word2id_file_name):
            return False
        # print("Pre-processed files exist. Loading them...")
        self.data_word = np.load(word_npy_file_name)
        self.data_pos1 = np.load(pos1_npy_file_name)
        self.data_pos2 = np.load(pos2_npy_file_name)
        self.data_mask = np.load(mask_npy_file_name)
        self.data_mask0 = np.load(mask0_npy_file_name)
        self.data_head = np.load(head_npy_file_name)
        self.data_tail = np.load(tail_npy_file_name)
        self.data_length = np.load(length_npy_file_name)
        self.rel2scope = json.load(open(rel2scope_file_name))
        self.word_vec_mat = np.load(word_vec_mat_file_name)
        self.word2id = json.load(open(word2id_file_name))
        if self.data_word.shape[1] != self.max_length:
            # print("Pre-processed files don't match current settings. Reprocessing...")
            return False
        # print("Finish loading")
        return True

    def __init__(self, file_name, word_vec_file_name, max_length=40, case_sensitive=False, reprocess=False, cuda=True):
        '''
        file_name: Json file storing the data in the following format
            {
                "P155": # relation id
                    [
                        {
                            "h": ["song for a future generation", "Q7561099", [[16, 17, ...]]], # head entity [word, id, location]
                            "t": ["whammy kiss", "Q7990594", [[11, 12]]], # tail entity [word, id, location]
                            "token": ["Hot", "Dance", "Club", ...], # sentence
                        },
                        ...
                    ],
                "P177":
                    [
                        ...
                    ]
                ...
            }
        word_vec_file_name: Json file storing word vectors in the following format
            [
                {'word': 'the', 'vec': [0.418, 0.24968, ...]},
                {'word': ',', 'vec': [0.013441, 0.23682, ...]},
                ...
            ]
        max_length: The length that all the sentences need to be extend to.
        case_sensitive: Whether the data processing is case-sensitive, default as False.
        reprocess: Do the pre-processing whether there exist pre-processed files, default as False.
        cuda: Use cuda or not, default as True.
        '''
        self.file_name = file_name
        self.word_vec_file_name = word_vec_file_name
        self.case_sensitive = case_sensitive
        self.max_length = max_length
        self.cuda = cuda

        if reprocess or not self._load_preprocessed_file():  # Try to load pre-processed files:
            # Check files
            if file_name is None or not os.path.isfile(file_name):
                raise Exception("[ERROR] Data file doesn't exist")
            if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
                raise Exception("[ERROR] Word vector file doesn't exist")

            # Load files
            print("Loading data file...")
            self.ori_data = json.load(open(self.file_name, "r"))
            print("Finish loading")
            print("Loading word vector file...")
            self.ori_word_vec = json.load(open(self.word_vec_file_name, "r"))
            print("Finish loading")

            # Eliminate case sensitive
            if not case_sensitive:
                print("Elimiating case sensitive problem...")
                for relation in self.ori_data:
                    for ins in self.ori_data[relation]:
                        for i in range(len(ins['tokens'])):
                            ins['tokens'][i] = ins['tokens'][i].lower()
                print("Finish eliminating")

            # Pre-process word vec
            self.word2id = {}
            self.word_vec_tot = len(self.ori_word_vec)  # the number of word features
            UNK = self.word_vec_tot  # the number of word features
            BLANK = self.word_vec_tot + 1  # the number of word features +1
            self.word_vec_dim = len(self.ori_word_vec[0]['vec'])  # the dim of word features
            print("Got {} words of {} dims".format(self.word_vec_tot, self.word_vec_dim))
            print("Building word vector matrix and mapping...")
            self.word_vec_mat = np.zeros((self.word_vec_tot, self.word_vec_dim), dtype=np.float32)
            for cur_id, word in enumerate(self.ori_word_vec):  # cur_id is the number of word
                w = word['word']  # every word and it's vec
                if not case_sensitive:
                    w = w.lower()
                self.word2id[w] = cur_id
                self.word_vec_mat[cur_id, :] = word['vec']
                self.word_vec_mat[cur_id] = self.word_vec_mat[cur_id] / np.sqrt(np.sum(self.word_vec_mat[cur_id] ** 2))
            self.word2id['UNK'] = UNK
            self.word2id['BLANK'] = BLANK
            print("Finish building")

            # Pre-process data
            print("Pre-processing data...")
            self.instance_tot = 0
            for relation in self.ori_data:
                self.instance_tot += len(self.ori_data[relation])
            self.data_word = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos1 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_pos2 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_head = np.zeros((self.instance_tot, 20), dtype=np.int32)
            self.data_tail = np.zeros((self.instance_tot, 20), dtype=np.int32)
            self.data_mask = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_mask0 = np.zeros((self.instance_tot, self.max_length), dtype=np.int32)
            self.data_length = np.zeros((self.instance_tot), dtype=np.int32)
            self.rel2scope = {}  # left close right open
            i = 0
            for relation in self.ori_data:
                self.rel2scope[relation] = [i, i]
                for ins in self.ori_data[relation]:
                    head = ins['h'][2][0]
                    head = list(head)
                    tail = ins['t'][2][0]
                    tail = list(tail)
                    pos1 = ins['h'][2][0][0]
                    pos2 = ins['t'][2][0][0]
                    words = ins['tokens']
                    cur_ref_data_word = self.data_word[i]
                    for j, word in enumerate(words):
                        if j < max_length:
                            if word in self.word2id:
                                cur_ref_data_word[j] = self.word2id[word]
                            else:
                                cur_ref_data_word[j] = UNK
                    for j in range(j + 1, max_length):
                        cur_ref_data_word[j] = BLANK
                    ##head
                    for h, h_word in enumerate(head):
                        if words[h_word] in self.word2id:
                            self.data_head[i][h] = self.word2id[words[h_word]]
                        else:
                            self.data_head[i][h] = UNK
                    for h in range(h + 1, 20):
                        self.data_head[i][h] = BLANK
                    ##tail
                    for t, t_word in enumerate(tail):
                        if words[t_word] in self.word2id:
                            self.data_tail[i][t] = self.word2id[words[t_word]]
                        else:
                            self.data_tail[i][t] = UNK
                    for t in range(t + 1, 20):
                        self.data_tail[i][t] = BLANK

                    self.data_length[i] = len(words)
                    if len(words) > max_length:
                        self.data_length[i] = max_length
                    if pos1 >= max_length:
                        pos1 = max_length - 1
                    if pos2 >= max_length:
                        pos2 = max_length - 1
                    pos_min = min(pos1, pos2)
                    pos_max = max(pos1, pos2)
                    for j in range(max_length):
                        self.data_pos1[i][j] = j - pos1 + max_length
                        self.data_pos2[i][j] = j - pos2 + max_length
                        if j >= self.data_length[i]:
                            self.data_mask[i][j] = 0
                        elif j <= pos_min:
                            self.data_mask[i][j] = 1
                        elif j <= pos_max:
                            self.data_mask[i][j] = 2
                        else:
                            self.data_mask[i][j] = 3
                        if j >= self.data_length[i]:
                            self.data_mask0[i][j] = 0
                        else:
                            self.data_mask0[i][j] = 1
                    i += 1
                self.rel2scope[relation][1] = i

            print("Finish pre-processing")

            print("Storing processed files...")
            name_prefix = '.'.join(file_name.split('/')[-1].split('.')[:-1])
            word_vec_name_prefix = '.'.join(word_vec_file_name.split('/')[-1].split('.')[:-1])
            processed_data_dir = '_processed_data'
            if not os.path.isdir(processed_data_dir):
                os.mkdir(processed_data_dir)
            np.save(os.path.join(processed_data_dir, name_prefix + '_word.npy'), self.data_word)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos1.npy'), self.data_pos1)
            np.save(os.path.join(processed_data_dir, name_prefix + '_pos2.npy'), self.data_pos2)
            np.save(os.path.join(processed_data_dir, name_prefix + '_head.npy'), self.data_head)
            np.save(os.path.join(processed_data_dir, name_prefix + '_tail.npy'), self.data_tail)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask.npy'), self.data_mask)
            np.save(os.path.join(processed_data_dir, name_prefix + '_mask0.npy'), self.data_mask0)
            np.save(os.path.join(processed_data_dir, name_prefix + '_length.npy'), self.data_length)
            json.dump(self.rel2scope, open(os.path.join(processed_data_dir, name_prefix + '_rel2scope.json'), 'w'))
            np.save(os.path.join(processed_data_dir, word_vec_name_prefix + '_mat.npy'), self.word_vec_mat)
            json.dump(self.word2id, open(os.path.join(processed_data_dir, word_vec_name_prefix + '_word2id.json'), 'w'))
            print("Finish storing")

    def next_one(self, N, K, Q, noise_rate=0):
        target_classes = random.sample(self.rel2scope.keys(), N)
        noise_classes = []
        for class_name in self.rel2scope.keys():
            if not (class_name in target_classes):
                noise_classes.append(class_name)
        # if len(noise_classes) ==0:
        #     noise_classes =['P706','p84','P495','P123','P57']
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'mask0': [], 'head': [], 'tail': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'mask0': [], 'head': [], 'tail': []}
        query_label = []

        for i, class_name in enumerate(target_classes):
            scope = self.rel2scope[class_name]
            indices = np.random.choice(list(range(scope[0], scope[1])), K + Q, False)
            word = self.data_word[indices]
            pos1 = self.data_pos1[indices]
            pos2 = self.data_pos2[indices]
            mask = self.data_mask[indices]
            mask0 = self.data_mask0[indices]
            head = self.data_head[indices]
            tail = self.data_tail[indices]
            support_word, query_word, _ = np.split(word, [K, K + Q])
            support_pos1, query_pos1, _ = np.split(pos1, [K, K + Q])
            support_pos2, query_pos2, _ = np.split(pos2, [K, K + Q])
            support_mask, query_mask, _ = np.split(mask, [K, K + Q])
            support_mask0, query_mask0, _ = np.split(mask0, [K, K + Q])
            support_head, query_head, _ = np.split(head, [K, K + Q])
            support_tail, query_tail, _ = np.split(tail, [K, K + Q])

            for j in range(K):
                prob = np.random.rand()
                if prob < noise_rate:
                    noise_class_name = noise_classes[np.random.randint(0, len(noise_classes))]
                    scope = self.rel2scope[noise_class_name]
                    indices = np.random.choice(list(range(scope[0], scope[1])), 1, False)
                    word = self.data_word[indices]
                    pos1 = self.data_pos1[indices]
                    pos2 = self.data_pos2[indices]
                    mask = self.data_mask[indices]
                    mask0 = self.data_mask0[indices]
                    head = self.data_head[indices]
                    tail = self.data_tail[indices]
                    support_word[j] = word
                    support_pos1[j] = pos1
                    support_pos2[j] = pos2
                    support_mask[j] = mask
                    support_mask0[j] = mask0
                    support_head[j] = head
                    support_tail[j] = tail

            support_set['word'].append(support_word)
            support_set['pos1'].append(support_pos1)
            support_set['pos2'].append(support_pos2)
            support_set['mask'].append(support_mask)
            support_set['mask0'].append(support_mask0)
            support_set['head'].append(support_head)
            support_set['tail'].append(support_tail)

            query_set['word'].append(query_word)
            query_set['pos1'].append(query_pos1)
            query_set['pos2'].append(query_pos2)
            query_set['mask'].append(query_mask)
            query_set['mask0'].append(query_mask0)
            query_set['head'].append(query_head)
            query_set['tail'].append(query_tail)
            query_label += [i] * Q

        support_set['word'] = np.stack(support_set['word'], 0)
        support_set['pos1'] = np.stack(support_set['pos1'], 0)
        support_set['pos2'] = np.stack(support_set['pos2'], 0)
        support_set['mask'] = np.stack(support_set['mask'], 0)
        support_set['mask0'] = np.stack(support_set['mask0'], 0)
        support_set['head'] = np.stack(support_set['head'], 0)
        support_set['tail'] = np.stack(support_set['tail'], 0)
        query_set['word'] = np.concatenate(query_set['word'], 0)
        query_set['pos1'] = np.concatenate(query_set['pos1'], 0)
        query_set['pos2'] = np.concatenate(query_set['pos2'], 0)
        query_set['mask'] = np.concatenate(query_set['mask'], 0)
        query_set['mask0'] = np.concatenate(query_set['mask0'], 0)
        query_set['head'] = np.concatenate(query_set['head'], 0)
        query_set['tail'] = np.concatenate(query_set['tail'], 0)
        query_label = np.array(query_label)

        perm = np.random.permutation(N * Q)
        query_set['word'] = query_set['word'][perm]
        query_set['pos1'] = query_set['pos1'][perm]
        query_set['pos2'] = query_set['pos2'][perm]
        query_set['mask'] = query_set['mask'][perm]
        query_set['mask0'] = query_set['mask0'][perm]
        query_set['head'] = query_set['head'][perm]
        query_set['tail'] = query_set['tail'][perm]
        query_label = query_label[perm]

        return support_set, query_set, query_label

    def get_sentence_adj(self, adj, sentence_num):
        # slice_user_graph = np.zeros((user_num,user_num), dtype=int)

        sentence_adj = np.zeros((sentence_num, sentence_num), dtype=int)
        A = adj

        A = np.matrix(A)
        A = sparse.coo_matrix(A)
        A = np.dot(A, A).todense()  # weights
        for i in range(sentence_num):
            for j in range(sentence_num):
                if (A[i, j] > 0):
                    sentence_adj[i][j] = 1
        # print('sentence adj de bian geshu ', sentence_adj.sum())
        return sentence_adj

    def get_entity_adj(self, adj, sentence_num, total_num):
        entity_num = total_num - sentence_num
        entity_adj = np.zeros((entity_num, entity_num), dtype=int)
        # print(adj.shape)
        A = adj

        A = np.matrix(A)
        A = sparse.coo_matrix(A)
        A = np.dot(A, A).todense()  # weights

        for i in range(sentence_num, total_num):
            for j in range(sentence_num, total_num):
                if (A[i, j] > 0):
                    entity_adj[i - sentence_num][j - sentence_num] = 1
        # print('entity adj de bian geshu ', entity_adj.sum())
        return entity_adj

    def adj_to_graph(self, adj):
        # print('##########adj_to_graph start')
        newG = nx.MultiGraph()
        for i in range(len(adj)):
            newG.add_node(i)
        # print('adj shape:', len(adj))

        for i in range(len(adj)):
            for j in range(len(adj)):
                if (adj[i][j] > 0):
                    newG.add_edge(i, j, weight=adj[i][j])
        # print('edge_num of thie graph:', newG.number_of_edges())
        # print('node_num of thie graph:', newG.number_of_nodes())
        return newG

    def array_to_matrix(self, adjs):
        f_adj = []
        for adj in adjs:
            a = sparse.csr_matrix(adj)
            f_adj.append(a)
        return f_adj

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sparse.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sparse.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sparse.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()  # D-1/2*A*D-1/2
        return adj_normalized

    def sen_wor_to_graph(self, B, N, K, Q, support_word, support_head, support_tail, query_word, query_head,
                         query_tail):
        # word_map = defaultdict(list)
        # head_map = defaultdict(list)
        # tail_map = defaultdict(list)
        # print(query_word)
        # print(support_word[0][0].__len__())
        # print(query_word[0][0].__len__())
        # input(x)

        class_num = B * N  # 2*5

        word_map = np.zeros((class_num, (Q + K * N), self.max_length), dtype=np.int32)
        head_map = np.zeros((class_num, (Q + K * N), support_head[0][0][0].size), dtype=np.int32)
        tail_map = np.zeros((class_num, (Q + K * N), support_head[0][0][0].size), dtype=np.int32)
        # print(word_map.shape)
        # print(head_map.shape)
        # print(tail_map.shape)

        word_num = B * N * (Q + K * N)
        head_num = B * N * (Q + K * N)
        tail_num = B * N * (Q + K * N)
        # print(word_num, head_num, tail_num)

        # map the sentence/entity with relation class
        for n in range(B):  # 5 support + 1 query
            # sentence
            a = support_word[n]  # 5 * 5 * 40
            b = query_word[n]  # 5 * 40
            for m in range(len(a)):  # a[m]: 5*40
                for j in range(a[m].shape[0] + 1):  # 5*40
                    if (j < 5):
                        word_map[n * N + m][j] = a[m][j]
                    else:
                        word_map[n * N + m][j] = b[m]

        for n in range(B):  # 5 support + 1 query
            # head_entity
            a = support_head[n]  # 5 * 5 * 20
            b = query_head[n]  # 5 * 20
            for m in range(len(a)):  # a[m]: 5*20
                for j in range(a[m].shape[0] + 1):  # 5*20
                    if (j < 5):
                        head_map[n * N + m][j] = a[m][j]
                    else:
                        head_map[n * N + m][j] = b[m]

        for n in range(B):  # 5 support + 1 query
            # tail_entity
            a = support_tail[n]  # 5 * 5 * 20
            b = query_tail[n]  # 5 * 20
            for m in range(len(a)):  # a[m]: 5*20
                for j in range(a[m].shape[0] + 1):  # 5*20
                    if (j < 5):
                        tail_map[n * N + m][j] = a[m][j]
                    else:
                        tail_map[n * N + m][j] = b[m]

        # entity_map = np.vstack((head_map, tail_map)) #120 * 20
        # entity_del_re = np.zeros(head_num + tail_num, dtype = int)
        #
        # for i in range(entity_map.shape[0]):
        #     entity_del_re[i] = i
        # for i in range(entity_map.shape[0]):
        #     for j in range(i+1, entity_map.shape[0]):
        #         if (entity_map[i] == entity_map[j]).all():
        #             print(i,j)
        #             entity_del_re[j] = entity_del_re[i]
        #             print("(((((((((((((((((((((((((((((((((")
        #
        # for i in range (head_map.__len__()):
        #     for j in range(head_map.__len__()):
        #         if (head_map[i] == head_map[j]).all():
        #             if i != j:
        #                 print("#######################44444444444444444444444")
        #                 print("head_map")
        #                 #print(head_map[i])
        #                 print(i,j)
        # for i in range (tail_map.__len__()):
        #     for j in range(tail_map.__len__()):
        #         if (tail_map[i] == tail_map[j]).all():
        #             if i!=j:
        #                 print("#######################44444444444444444444444")
        #                 print("tail_map")
        #                 #print(tail_map[i])
        #                 print(i, j)
        # for i in range (head_map.__len__()):
        #     for j in range(tail_map.__len__()):
        #         if (head_map[i] == tail_map[j]).all():
        #             print("#######################44444444444444444444444")
        #             print("head_map and tail_map")
        #             #print(head_map)
        #             print(i, j)
        # print(entity_del_re)
        # entity_num = (np.unique(entity_del_re)).shape
        # print(entity_num)
        num = word_num + head_num + tail_num
        adj = np.zeros((class_num, 3 * (1 + K * N + N), 3 * (1 + K * N + N)), dtype=np.int32)  # 1 query + K * N support
        add_adj = np.zeros((class_num, 3 * (1 + K * N + N), 3 * (1 + K * N + N)), dtype=np.int32)

        #(26+5)*3 N:class num K:support num per one class
        for i in range (class_num):
            for j in range(3 * (1 + K * N + N)):
                for k in range(3 * (1 + K * N + N)):
                    adj[i][j][k] = 1
            #自连接
            for j in range(3 * (1 + K * N + N)):
                add_adj[i][j][j] = 1
            #句子和头尾实体连接
            for j in range(1 + K * N + N):
                add_adj[i][j][1 + K * N + N + j] = 1
                add_adj[i][1 + K * N + N + j][j] = 1
                add_adj[i][j][(1 + K * N + N) * 2 + j] = 1
                add_adj[i][(1 + K * N + N) * 2 + j][j] = 1
                add_adj[i][1 + K * N + N + j][(1 + K * N + N) * 2 + j] = 1
                add_adj[i][(1 + K * N + N) * 2 + j][1 + K * N + N + j] = 1

            #每个类别的支持集节点与”超级节点“（在每个类别的最后一个索引）连接

            #句子节点与”超级节点“建立连接
            for l in range(1, N + 1):
                p = 1 + l * K + (l - 1)
                for j in range(p - K, p):
                    add_adj[i][j][p] = 1
                    add_adj[i][p][j] = 1
            #头实体节点与”超级节点“建立连接
            for l in range(1, N + 1):
                start = 1 + K * N + N
                p = start + (1 + l * K + (l - 1))
                for j in range(p - K, p):
                    add_adj[i][j][p] = 1
                    add_adj[i][p][j] = 1
            #尾实体节点与”超级节点“建立连接
            for l in range(1, N + 1):
                start = (1 + K * N + N) * 2
                p = start + (1 + l * K + (l - 1))
                for j in range(p - K, p):
                    add_adj[i][j][p] = 1
                    add_adj[i][p][j] = 1

            #”超级节点“与query节点建立连接
            for l in range(1, N + 1):
                p = 1 + l * K + (l - 1)
                j = 0
                add_adj[i][p][j] = 1
                add_adj[i][j][p] = 1
            for l in range(1, N + 1):
                start = 1 + K * N + N
                p = start + (1 + l * K + (l - 1))
                j = start + 1
                add_adj[i][p][j] = 1
                add_adj[i][j][p] = 1
            for l in range(1, N + 1):
                start = (1 + K * N + N) * 2
                p = start + (1 + l * K + (l - 1))
                j = start + 1
                add_adj[i][p][j] = 1
                add_adj[i][j][p] = 1
        adj = adj + add_adj

        # 全连接邻接矩阵
        # for i in range(class_num):
        #     for j in range(3 * (1 + K * N)):
        #         for k in range(3 * (1 + K * N)):
        #             adj[i][j][k] = 1
        # for i in range(class_num):
        #     for j in range(1 + K * N):
        #         add_adj[i][j][j] = 1
        #         add_adj[i][j][1 + K * N + j] = 1
        #         add_adj[i][1 + K * N + j][j] = 1
        #         add_adj[i][j][2 * (1 + K * N) + j] = 1
        #         add_adj[i][2 * (1 + K * N) + j][j] = 1
        #         add_adj[i][2 * (1 + K * N) + j][1 + K * N + j] = 1
        #         add_adj[i][1 + K * N + j][2 * (1 + K * N) + j] = 1
        # for l in range(1, 6):
        #     for j in range(1 + (l - 1) * N, 1 + l * N):
        #         for k in range(1 + (l - 1) * N, 1 + l * N):
        #             add_adj[i][j][k] = 1
        #             add_adj[i][k][j] = 1
        # adj = adj + add_adj

        # 稀疏邻接矩阵
        # wei = 0.2
        # for i in range(class_num):
        #     for j in range(3 * (1 + K * N)):
        #         for k in range(3 * (1 + K * N)):
        #             adj[i][j][k] = 1/(3 * (1 + K * N))
        # for i in range(class_num):
        #     for j in range(1 + K * N):
        #         add_adj[i][j][j] = wei
        #         add_adj[i][j][1 + K * N + j] = wei
        #         add_adj[i][1 + K * N + j][j] = wei
        #         add_adj[i][j][2 * (1 + K * N) + j] = wei
        #         add_adj[i][2 * (1 + K * N) + j][j] = wei
        #         add_adj[i][2 * (1 + K * N) + j][1 + K * N + j] = wei
        #         add_adj[i][1 + K * N + j][2 * (1 + K * N) + j] = wei
        # adj = adj + add_adj

        sen_adj = []
        entity_adj = []
        sen_graph = []
        entity_graph = []
        total_graph = []
        for i in range(class_num):
            sen_adj.append(self.get_sentence_adj(adj[i], 1 + K * N))
            entity_adj.append(self.get_entity_adj(adj[i], 1 + K * N, 3 * (1 + K * N)))
        for i in range(class_num):
            sen_graph.append(self.adj_to_graph(sen_adj[i]))
            entity_graph.append(self.adj_to_graph(entity_adj[i]))
            total_graph = self.adj_to_graph(adj[i])

        np.set_printoptions(threshold=sys.maxsize)
        # with open('test.txt', 'w') as f:
        #     print(adj, file=f)
        # print(type(adj))
        adj = self.array_to_matrix(adj)

        adj = [self._normalize_graph_gcn(a) for a in adj]

        return adj, sen_adj, entity_adj, total_graph, sen_graph, entity_graph

    def next_batch(self, B, N, K, Q, noise_rate=0):
        support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'mask0': [], 'head': [], 'tail': []}
        query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'mask0': [], 'head': [], 'tail': []}
        label = []
        for one_sample in range(B):
            current_support, current_query, current_label = self.next_one(N, K, Q, noise_rate=noise_rate)
            support['word'].append(current_support['word'])
            support['pos1'].append(current_support['pos1'])
            support['pos2'].append(current_support['pos2'])
            support['mask'].append(current_support['mask'])
            support['mask0'].append(current_support['mask0'])
            support['head'].append(current_support['head'])
            support['tail'].append(current_support['tail'])
            query['word'].append(current_query['word'])
            query['pos1'].append(current_query['pos1'])
            query['pos2'].append(current_query['pos2'])
            query['mask'].append(current_query['mask'])
            query['mask0'].append(current_query['mask0'])
            query['head'].append(current_query['head'])
            query['tail'].append(current_query['tail'])
            label.append(current_label)
        #adj, sen_adj, entity_adj, total_graph, sen_graph, entity_graph = self.sen_wor_to_graph(B, N, K, Q,support['word'],support['head'],support['tail'],query['word'],query['head'],query['tail'])
        adj =[]
        sen_adj = []
        entity_adj = []
        total_graph = []
        sen_graph = []
        entity_graph = []

        support['word'] = Variable(torch.from_numpy(np.stack(support['word'], 0)).long().view(-1, self.max_length))
        support['pos1'] = Variable(torch.from_numpy(np.stack(support['pos1'], 0)).long().view(-1, self.max_length))
        support['pos2'] = Variable(torch.from_numpy(np.stack(support['pos2'], 0)).long().view(-1, self.max_length))
        support['mask'] = Variable(torch.from_numpy(np.stack(support['mask'], 0)).long().view(-1, self.max_length))
        support['mask0'] = Variable(torch.from_numpy(np.stack(support['mask0'], 0)).long().view(-1, self.max_length))
        support['head'] = Variable(torch.from_numpy(np.stack(support['head'], 0)).long().view(-1, 20))
        support['tail'] = Variable(torch.from_numpy(np.stack(support['tail'], 0)).long().view(-1, 20))
        query['word'] = Variable(torch.from_numpy(np.stack(query['word'], 0)).long().view(-1, self.max_length))
        query['pos1'] = Variable(torch.from_numpy(np.stack(query['pos1'], 0)).long().view(-1, self.max_length))
        query['pos2'] = Variable(torch.from_numpy(np.stack(query['pos2'], 0)).long().view(-1, self.max_length))
        query['mask'] = Variable(torch.from_numpy(np.stack(query['mask'], 0)).long().view(-1, self.max_length))
        query['mask0'] = Variable(torch.from_numpy(np.stack(query['mask0'], 0)).long().view(-1, self.max_length))
        query['head'] = Variable(torch.from_numpy(np.stack(query['head'], 0)).long().view(-1, 20))
        query['tail'] = Variable(torch.from_numpy(np.stack(query['tail'], 0)).long().view(-1, 20))
        label = Variable(torch.from_numpy(np.stack(label, 0).astype(np.int64)).long())

        # To cuda
        if self.cuda:
            for key in support:
                support[key] = support[key].cuda()
            for key in query:
                query[key] = query[key].cuda()
            label = label.cuda()
        # print("9999999999999999999999000000000000000000000")
        # print(support["word"].__len__())
        # print("9999999999999999999999000000000000000000000")
        # input(x)
        return support, query, label

    # new
    def lookup(self, ins):
        head = ins['h'][2][0]
        head = list(head)
        tail = ins['t'][2][0]
        tail = list(tail)
        pos1 = ins['h'][2][0][0]
        pos2 = ins['t'][2][0][0]
        words = ins['tokens']
        cur_ref_data_word = np.zeros(self.max_length, dtype=np.int32)
        data_pos1 = np.zeros(self.max_length, dtype=np.int32)
        data_pos2 = np.zeros(self.max_length, dtype=np.int32)
        data_mask = np.zeros(self.max_length, dtype=np.int32)
        data_mask0 = np.zeros(self.max_length, dtype=np.int32)
        data_head = np.zeros(20, dtype=np.int32)
        data_tail = np.zeros(20, dtype=np.int32)
        for j, word in enumerate(words):
            word = word.lower()
            if j < self.max_length:
                if word in self.word2id:
                    cur_ref_data_word[j] = self.word2id[word]
                else:
                    cur_ref_data_word[j] = self.word2id['UNK']
        for j in range(j + 1, self.max_length):
            cur_ref_data_word[j] = self.word2id['BLANK']

        ##head
        for h, h_word in enumerate(head):
            if words[h_word] in self.word2id:
                data_head[h] = self.word2id[words[h_word]]
            else:
                data_head[h] = self.word2id['UNK']
        for h in range(h + 1, 20):
            data_head[h] = self.word2id['BLANK']
        ##tail
        for t, t_word in enumerate(tail):
            if words[t_word] in self.word2id:
                data_tail[t] = self.word2id[words[t_word]]
            else:
                data_tail[t] = self.word2id['UNK']
        for t in range(t + 1, 20):
            data_tail[t] = self.word2id['BLANK']

        data_length = len(words)
        if len(words) > self.max_length:
            data_length = self.max_length
        if pos1 >= self.max_length:
            pos1 = self.max_length - 1
        if pos2 >= self.max_length:
            pos2 = self.max_length - 1
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(self.max_length):
            data_pos1[j] = j - pos1 + self.max_length
            data_pos2[j] = j - pos2 + self.max_length
            if j >= data_length:
                data_mask[j] = 0
            elif j <= pos_min:
                data_mask[j] = 1
            elif j <= pos_max:
                data_mask[j] = 2
            else:
                data_mask[j] = 3
            if j >= data_length:
                data_mask0[j] = 0
            else:
                data_mask0[j] = 1

        return cur_ref_data_word, data_pos1, data_pos2, data_mask, data_mask0, data_head, data_tail, data_length
