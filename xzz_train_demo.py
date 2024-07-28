import models
from fewshot_re_kit.data_g_loader import JSONFileDataLoader
from fewshot_re_kit.xzz_framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder
# from models.proto import Proto
# from models.proto_adv import Proto_adv
# from models.gnn_adv import GNN_adv
from models.my_hen_gnn_adv import hen_GNN_adv
from models.my_hen_gnn_noadv import hen_GNN_noadv
from models.my_hen_gat_noadv import hen_GAT_noadv
from models.my_hen_gat_adv import hen_GAT_adv
from models.agnn import GNN
from models.agnn_adv import GNN_adv
from models.agnn_second import GNN_second
from models.agnn_three import GNN_three
from models.gnn import GNN_
# from models.snail import SNAIL
# from models.metanet import MetaNet
import argparse

import sys
from torch import optim
import torch
torch.cuda.set_device(-1)

parser = argparse.ArgumentParser()


model_name = 'agnn_three'
N = 5
K = 5
noise_rate = 0

if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    N = int(sys.argv[2])
if len(sys.argv) > 3:
    K = int(sys.argv[3])
if len(sys.argv) > 4:
    noise_rate = float(sys.argv[4])

print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
print("Model: {}".format(model_name))

max_length = 40
train_data_loader = JSONFileDataLoader('./my_data_noisy/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./my_data_noisy/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./my_data_noisy/test.json', './data/glove.6B.50d.json', max_length=max_length)

framework = FewShotREFramework(train_data_loader, val_data_loader,test_data_loader)
sentence_encoder = CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)

if model_name == 'proto':
    model = Proto(sentence_encoder)
    framework.train(model, model_name, 2, N, N, K, 1)
elif model_name == 'proto_adv':
    model = Proto_adv(sentence_encoder)
    framework.train(model, model_name, 4, 20, N, K, 5)
elif model_name == 'gnn':
    print("noise_rate:", noise_rate)
    model = GNN_(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'gnn_adv':
    print("noise_rate:" ,noise_rate)
    model = GNN_adv(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'hen_gnn_adv':
    print("noise_rate:" ,noise_rate)
    model = hen_GNN_adv(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'hen_gnn_noadv':
    print("noise_rate:" ,noise_rate)
    model = hen_GNN_noadv(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'hen_gat_noadv':
    print("noise_rate:" ,noise_rate)
    model = hen_GAT_noadv(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'hen_gat_adv':
    print("noise_rate:" ,noise_rate)
    model = hen_GAT_adv(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'agnn':
    print("noise_rate:" ,noise_rate)
    model = GNN(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'agnn_adv':
    print("noise_rate:" ,noise_rate)
    model = GNN(sentence_encoder, N)
    framework.train(model, model_name, 2, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'agnn_second':
    print("noise_rate:" ,noise_rate)
    model = GNN_second(sentence_encoder, N)
    framework.train(model, model_name, 1, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)
elif model_name == 'agnn_three':
    print("noise_rate:" ,noise_rate)
    model = GNN_three(sentence_encoder, N)
    framework.train(model, model_name, 4, N, N, K, 1, learning_rate=5e-4, weight_decay=0, optimizer=optim.Adam, noise_rate=noise_rate)

elif model_name == 'snail':
    print("HINT: SNAIL works only in PyTorch 0.3.1")
    model = SNAIL(sentence_encoder, N, K)
    framework.train(model, model_name, 25, N, N, K, 1, learning_rate=1e-2, weight_decay=0, optimizer=optim.SGD)
elif model_name == 'metanet':
    model = MetaNet(N, K, train_data_loader.word_vec_mat, max_length)
    framework.train(model, model_name, 1, N, N, K, 1, learning_rate=5e-3, weight_decay=0, optimizer=optim.Adam, train_iter=300000)
else:
    raise NotImplementedError

