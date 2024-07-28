import os
# import sklearn.metrics
import numpy as np
from scipy import spatial
import sys
import time
from . import sentence_encoder
from . import data_g_loader
import torch
from torch import autograd, optim, nn

from torch.autograd import Variable
from torch.nn import functional as F

class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.CrossEntropyLoss()
        
    
    def forward(self, support, query, N, K, Q, label, model, is_training):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError
        
    def gaussian(x, y, sigma=1.0):
        distance = torch.norm(x-y)
        similaraty = np.exp(-distance**2/(2*(sigma **2)))
        return similaraty

    def loss(self, logits, label, B, N, K):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        # N = logits.size(-1)
        # return self.cost(logits.view(-1, N), label.view(-1))

        logits = logits.contiguous()
        N = logits.size(-1)
        B=label.view(-1).contiguous()
        A=logits.view(-1, N)
        cost_loss = self.cost(A,B)
        #C_num = out_fea.shape[0]
        #fea_ave = torch.zeros(C_num, K, N).cuda()#10*5*5
        #fea_loss = torch.zeros(C_num, K).cuda()#10*5
        loss = cost_loss
        
        off = -1
        
        '''
        if off==0:
            for i in range(C_num):
                for k in range(K):
                    #print(out_fea[0:2])
                    #print(out_fea[i][1 + k * N : 1 + k * N + K])
                    #print(out_fea[i][1 + k * N : 1 + k * N + K].mean(0))
                    #input(x)
                    fea_ave[i][k] = out_fea[i][1+k*N : 1+k*N+K].mean(0)
            
            for i in range(C_num):
                for k in range(K):
                    sum = 0
                    for j in range(1+k*N, 1+k*N+K):
                        #print(j)
                        #print(fea_ave[i][k])
                        #print(out_fea[i][j])
                        
                        #sum += 1-spatial.distance.cosine(fea_ave[i][k].data.cpu().numpy(), out_fea[i][j].data.cpu().numpy())
                        sum += 1- self.gaussian(fea_ave[i][k].data.cpu().numpy(), out_fea[i][j].data.cpu().numpy())
                    fea_loss[i][k] = sum
        
            #print(fea_loss)
            #print(fea_loss.sum())
            #print(fea_loss.mean())
            #input(x)
            loss = fea_loss.sum() + cost_loss
        #相同类别的向量做平均求相似度
        if off==1:
            fea_ave_second = torch.zeros(C_num, K, N).cuda()#10*5*5
            for i in range(C_num):
                for k in range(K):   
                    fea_ave[i][k] = out_fea[i][1+k*N : 1+k*N+K].mean(0)
            
            for i in range(0,5):
                for k in range(K):
                    fea_ave_second[i][k] = fea_ave[0:5, k, :].mean(0)
            for i in range(5,10):
                for k in range(K):
                    fea_ave_second[i][k] = fea_ave[5:10, k, :].mean(0)
     
            
            for i in range(C_num):
                for k in range(K):
                    sum = 0
                    for j in range(1+k*N, 1+k*N+K):          
                        sum += 1 - spatial.distance.cosine(fea_ave_second[i][k].data.cpu().numpy(), out_fea[i][j].data.cpu().numpy())
                    fea_loss[i][k] = sum
  
            loss = fea_loss.sum() + cost_loss
        '''
        
        
        
        return loss

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    
class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader,test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            # checkpoint = torch.load(ckpt,map_location='cpu')
            checkpoint = torch.load(ckpt)
            # print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              ckpt_dir='./checkpoint',
              test_result_dir='./test_result',
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=240000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              cuda=True,
              # pretrain_model='./checkpoint/gnn.pth.tar',
              pretrain_model=None,
              optimizer=optim.SGD,
              noise_rate=0):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")
        
        # Init
        parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            # start_iter = checkpoint['iter'] + 1
            start_iter = 12000
        else:
            start_iter = 0

        if cuda:
            model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        not_best_count = 0 # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            support, query, label = self.train_data_loader.next_batch(B, N_for_train, K, Q, noise_rate=noise_rate)
            logits, pred = model(support, query, B, N_for_train, K, Q ,label, model, it, 0, is_training=True)
            
            # print(logits)
            # print(pred)
            # print(label)
            # input(ooo)
            # print(logits)
            # input(fram)
            
            loss = model.loss(logits, label, B, N_for_train, K)
            right = model.accuracy(pred, label)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(parameters_to_optimize, 10)
            optimizer.step()
            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            

            # if iter_sample==2:
            #     input(t)
            
            sys.stdout.write('step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_right = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, K, Q, val_iter, noise_rate=noise_rate)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name+str(N_for_eval)+ '-'+ str(K) + '.pth.tar')
                    torch.save({'state_dict': model.state_dict(), 'iters': it + 1,
                                'best_acc': acc, 'optimizer': optimizer.state_dict()}
                               , save_path)

                    best_acc = acc
                
        print("\n####################\n")
        print("Finish training " + model_name)
        test_acc = self.eval(model, B, N_for_eval, K, Q, test_iter, ckpt=os.path.join(ckpt_dir, model_name+str(N_for_eval)+ '-'+ str(K) + '.pth.tar'), noise_rate=noise_rate)
        # test_acc = self.eval(model, B, N_for_eval, K, Q, test_iter, ckpt=None)

        print("Test accuracy: {}".format(test_acc))

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None,
             noise_rate=0):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        for it in range(eval_iter):
            support, query, label = eval_dataset.next_batch(B, N, K, Q, noise_rate=noise_rate)
            logits, pred= model(support, query, B, N, K, Q ,label, model, it, eval_iter, is_training=False )
            right = model.accuracy(pred, label)
            iter_right += self.item(right.data)
            iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | accuracy: {1:3.2f}%'.format(it + 1, 100 * iter_right / iter_sample) +'\r')
            sys.stdout.flush()
        print("")
        return iter_right / iter_sample

    # new
    def set_model(self, model, ckpt = None):
        checkpoint = self.__load_model__(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        self.model = model

    def predict(self, support, query, N, K, Q,label=None, model=None ,is_training=False):
        return self.model(support, query, N, K, Q,label=None, model=None,is_training=False)