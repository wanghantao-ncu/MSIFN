
import os
import shutil
import time
import pprint
import torch
import numpy as np
import network.resnet as resnet
import os.path as osp
import random
import torch.nn.functional as F

model_dict = dict(
    ResNet10=resnet.ResNet10,
    ResNet12=resnet.ResNet12,
    ResNet18=resnet.ResNet18,
    ResNet34=resnet.ResNet34,
    ResNet34s=resnet.ResNet34s,
    ResNet50=resnet.ResNet50,
    ResNet101=resnet.ResNet101,
    WRN28=resnet.WRN28
    )

# 设置随机种子来控制随机数生成过程，从而确保程序的可重复性
def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

# 从指定路径加载一个保存的模型状态字典，并将其参数更新到当前模型中
def load_model(model, dir):
    model_dict = model.state_dict()
    file_dict = torch.load(dir)['state']
    for k, v in file_dict.items():
        if k not in model_dict:
            print(k)
    file_dict = {k: v for k, v in file_dict.items() if k in model_dict}
    model_dict.update(file_dict)
    model.load_state_dict(model_dict)
    return model

# 计算局部加权值
def compute_weight_local(feat_g,feat_ql,feat_sl,temperature=2.0):
    # feat_g : nk * dim
    # feat_l : nk * m * dim
    [_,k,m,dim] = feat_sl.shape
    [n,q,m,dim] = feat_ql.shape

    feat_g_expand = feat_g.unsqueeze(2).expand_as(feat_ql)
    sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql,dim=-1)
    I_opp_m = (1 - torch.eye(m)).unsqueeze(0).to(sim_gl.device)
    sim_gl = -(torch.matmul(sim_gl, I_opp_m).unsqueeze(-2))/(m-1)


    return sim_gl

#  proto_walk
def compute_weight_local(feat_g, feat_ql,feat_sl,measure = "cosine"):
    # feat_g : nk * dim
    # feat_l : nk * m * dim
    # [_,k,m,dim] = feat_sl.shape
    # [n,q,m,dim] = feat_ql.shape
    # # print(feat_ql.shape)
    # # feat_g_expand --miniImageNet:(1,1,n_way,1,feature_dim)
    # feat_g_expand = torch.mean(feat_g,dim=1).unsqueeze(0).unsqueeze(1).unsqueeze(3)
    # if measure == "cosine":
    #     # sim_gl --miniImageNet:(5,5,5,16)--(1,1,n_way,1,feature_dim)和(n_way,n_shot,1,idx_walk,feature_dim)进行cos相似度计算的(n_way,n_shot,n_way,idx_walk)
    #     sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql.unsqueeze(2),dim=-1)
    # else:
    #     sim_gl = -1 * 0.002 * torch.sum((feat_g_expand - feat_ql.unsqueeze(2)) ** 2, dim=-1)
    # # I_m (1,1,idx_walk,idx_walk)--(1,1,16,16)
    # I_m = torch.eye(m).unsqueeze(0).unsqueeze(1).to(sim_gl.device)
    # sim_gl =  torch.matmul(sim_gl, I_m)

    [_,k,m,dim] = feat_ql.shape
    feat_g_expand = torch.mean(feat_g,dim=1).unsqueeze(1).unsqueeze(1)
    sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql,dim=-1)
    # if k == 15:
    #     feat_g_expand = torch.mean(feat_g,dim=1).unsqueeze(0).unsqueeze(1).unsqueeze(3)
    #         # sim_gl --miniImageNet:(5,5,5,16)--(1,1,n_way,1,feature_dim)和(n_way,n_shot,1,idx_walk,feature_dim)进行cos相似度计算的(n_way,n_shot,n_way,idx_walk)
    #     sim_gl = torch.cosine_similarity(feat_g_expand,feat_ql.unsqueeze(2),dim=-1)
    return sim_gl

def feature_fusion_local(proto_moving, feat_g, feat_sl, k):
    [_,l,m,dim] = feat_sl.shape
    proto_moving_expand = proto_moving.unsqueeze(1).unsqueeze(1)
    # feat_sl = feat_sl.reshape(feat_sl.size(0),-1,feat_sl.size(3))
    matrix_L2_dist = torch.linalg.norm(proto_moving_expand - feat_sl, dim=-1)
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices
    weight = torch.div(1, 1 + matrix_L2_dist)
    gather_weight = torch.gather(weight, dim=-1, index=index).unsqueeze(-1)
    gather_feat_sl = torch.gather(feat_sl, dim=-2, index=index.unsqueeze(-1).expand(-1, -1, k, dim))

    Weight_gathered_local = torch.matmul(gather_feat_sl.permute(0, 1, 3, 2), gather_weight).squeeze(-1)
    feature_fusion_local = torch.div((Weight_gathered_local + feat_g),
                                        torch.sum(weight, dim=2).unsqueeze(-1) + 1)

    return feature_fusion_local

def base_feature_gather(proto_moving, base_means, sem_n1, emb_base, k):
    n_way, n_dim = proto_moving.shape
    n_classes = base_means.shape[0]
    assert base_means.shape == (n_classes, n_dim)

    base_means = base_means.unsqueeze(0).expand(n_way, n_classes, n_dim)
    matrix_L2_dist = torch.linalg.norm(sem_n1.unsqueeze(1) - emb_base.unsqueeze(0), dim=-1)
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices
    gathered_base = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(n_way, k, n_dim))
    assert gathered_base.shape == (n_way, k, n_dim)

    return gathered_base

def feature_fusion_base_local(proto_moving, feature_local_fusion, base_means, sem_n1, emb_base, k):
    n_way, n_dim = proto_moving.shape
    proto_moving_reshaped = proto_moving.reshape(n_way, 1, n_dim)
    n_classes = base_means.shape[0]
    base_means = base_means.unsqueeze(0).expand(n_way, n_classes, n_dim)
    #选择基类
    # matrix_L2_dist_sem = torch.linalg.norm(sem_n1.unsqueeze(1) - emb_base.unsqueeze(0), dim=-1)
    # index = torch.topk(matrix_L2_dist_sem, k, dim=-1, largest=False, sorted=True).indices
    matrix_L2_dist = torch.linalg.norm(proto_moving_reshaped - base_means, dim=-1)
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices

    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(n_way, k, n_dim))#(n_way,k,feature_dim)
    #计算基类原型权重
    matrix_L2_dist_base = torch.linalg.norm(proto_moving.unsqueeze(1) - gathered_mean, dim=-1)
    weight_base = torch.div(1, 1 + matrix_L2_dist_base)#(n_way,k)
    #计算局部特征校准原型权重
    matrix_L2_dist_local = torch.linalg.norm(proto_moving.unsqueeze(1) - feature_local_fusion, dim=-1)
    weight_local = torch.div(1, 1 + matrix_L2_dist_local)

    Weight_gathered_mean_base = torch.matmul(gathered_mean.permute(0, 2, 1), weight_base.unsqueeze(-1)).squeeze(-1)
    Weight_gathered_mean_local = torch.matmul(feature_local_fusion.permute(0, 2, 1), weight_local.unsqueeze(-1)).squeeze(-1)

    feature_fusion = torch.div((Weight_gathered_mean_base + Weight_gathered_mean_local + proto_moving),
                               torch.sum(weight_base, dim=1).unsqueeze(-1) + torch.sum(weight_local, dim=1).unsqueeze(-1) + 1)
    return feature_fusion

def feature_fusion_base(proto_moving, base_means, k):
    n_way, n_dim = proto_moving.shape
    batch_dim = n_way
    proto_moving_reshaped = proto_moving.reshape(batch_dim, 1, n_dim)

    n_classes = base_means.shape[0]
    base_means = base_means.unsqueeze(0).expand(batch_dim, n_classes, n_dim)

    matrix_L2_dist = torch.linalg.norm(proto_moving_reshaped - base_means, dim=-1)
    index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices

    Weight = torch.div(1, 1 + matrix_L2_dist)

    gather_weight = torch.gather(Weight, dim=-1, index=index).unsqueeze(-1).reshape(batch_dim, k, 1)

    gathered_mean = torch.gather(base_means, dim=-2, index=index.unsqueeze(-1).expand(batch_dim, k, n_dim))

    Weight_gathered_mean = torch.matmul(gathered_mean.permute(0, 2, 1), gather_weight).reshape(batch_dim, n_dim)

    proto_moving = torch.div((Weight_gathered_mean + proto_moving_reshaped.reshape(batch_dim, n_dim)),
                            torch.sum(gather_weight, dim=1) + 1)
    return proto_moving

def tc_proto(feat, label, way):
    feat_proto = torch.zeros(way, feat.size(1))
    for lb in torch.unique(label):
        ds = torch.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = torch.mean(feat_, dim=0)
    if torch.cuda.is_available():
        feat_proto = feat_proto.type(feat.type())
    return feat_proto

def compactness_loss(gen_feat, gen_label, proto, supp_label, n_sift_aug):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    loss = 0
    num = n_sift_aug
    for lb in torch.unique(gen_label):
        for id in range(lb*num,lb*num+num):
            loss = loss + loss_fn(gen_feat[id], proto[lb])
    return loss

if __name__ == '__main__':
    feat_g = torch.randn((5,15,64))
    # feat_g = torch.ones((5,3,64))
    feat_sl = torch.randn((5,3,6,64))
    feat_ql = torch.randn((5,15,6,64))
    # feat_l = torch.ones((5,3,6,64))
    compute_weight_local(feat_g,feat_ql,feat_sl)
    # print(compute_weight_local(feat_g,feat_ql,feat_sl)[0,0])