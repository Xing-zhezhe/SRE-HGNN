import torch
import numpy as np
#针对gnn
def save_x( x, label, name, NQ ):

    if name == 'beginning':
        kz = x.shape[1]
        px_query = x.view(-1, kz)
    else:  
        pxx = x.split(26, 1)[0]#按照第二个维度均分，每份26个，[0]指第一份的内容 10*26*230
        px = pxx[:, 0, :]#取query实例信息 (1*)10*230
        ps = pxx[:, 1:, :].split(NQ, 0)#support实例信息平均分配，每份NQ个，即每个batch的query个数，此时为5 ,且是从第二个向量开始数，维度：5*25*230两个
        ps1 = ps[0]#第一个batch的support 5*25*230
        ps2 = ps[1]#第二个batch的support 5*25*230
        z1 = ps1.shape[2]#230
        z2 = ps2.shape[2]#230
        ps1 = ps1[0, :, :].view(-1, z1)#25*230
        ps2 = ps2[0, :, :].view(-1, z2)#25*230
        pss = torch.cat([ps1, ps2], 0)#50个样本合并 50*230
        px = torch.cat([pss, px], 0)#60*230 先support后query

        if name == 'before':
            slabel = pss[:, -5:]
            _, slabel = torch.max(slabel, 1)
            label = torch.cat([slabel.view(-1), label.view(-1)], 0)
            px_query = px[:, :-5]
            px_query = px_query.view(-1, 230)
        elif name == 'After CNN' or name == 'GNN_1'or name == 'GNN_2'or name == 'GNN_3'or name == 'GNN_4'or name == 'end':
            kz = px.shape[1]
            px_query = px.view(-1, kz)
        else:
            return False
    px_query = px_query.data.cpu().numpy()
    label = label.data.cpu().numpy()

    name_ = './data_image/xzz/{}_data'.format(name)
    np.savez(name_, px_query, label)

    return True