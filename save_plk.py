from __future__ import print_function

import argparse
import csv
import os
import collections
import pickle
import random

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from io_utils import parse_args
from data.datamgr import SimpleDataManager , SetDataManager
from methods.template import BaselineTrain
from utils.utils import *

import network.resnet

import torch.nn.functional as F

from io_utils import parse_args, get_resume_file ,get_assigned_file
from os import path

use_gpu = torch.cuda.is_available()

# 封装其他的PyTorch模型
class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module 
    def forward(self, x):
        return self.module(x)
# 将给定的数据（Python对象）序列化后保存到指定的文件中
def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
# 从指定的文件中读取并反序列化存储的数据
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def extract_feature(val_loader, model, checkpoint_dir, tag='last',set='base'):
    # 创建一个保存目录路径
    save_dir = '{}/{}'.format(checkpoint_dir, tag)
    # 如果保存目录中已经存在特征文件，则直接读取并返回，否则创建该目录
    if os.path.isfile(save_dir + '/%s_features.plk'%set):
        data = load_pickle(save_dir + '/%s_features.plk'%set)
        return data
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    #model.eval()
    with torch.no_grad():
        # 初始化默认字典output_dict
        output_dict = collections.defaultdict(list)
        # 遍历val_loader中所有输入和标签
        for i, (inputs, labels) in enumerate(val_loader):
            # compute output
            # 将输入数据移动到GPU上
            inputs = inputs.cuda()
            labels = labels.cuda()
            # 将输入通过深度学习模型进行推理，得到输出特征
            outputs = model.feature_forward(inputs)
            # 将输出特征从GPU上移回CPU上并转换为NumPy数组
            outputs = outputs.cpu().data.numpy()
            
            # 将输出特征和标签分别存入output_dict中
            for out, label in zip(outputs, labels):
                output_dict[label.item()].append(out)

        # 将提取到的特征存入文件中
        all_info = output_dict
        save_pickle(save_dir + '/%s_features.plk'%set, all_info)
        return all_info

if __name__ == '__main__':
    params = parse_args()
    if params.dataset == 'cub':
        params.model = 'ResNet18'
    else:
        params.model = 'ResNet12'
    params.method = 'stl_deepbdc'

    json_file_read = False
    if params.dataset == 'miniimagenet':
        base_file = 'train'
        novel_file = 'test'
        params.num_classes = 64
    elif params.dataset == 'cub':
        base_file = 'base.json'
        novel_file =  'novel.json'
        params.num_classes = 200
        json_file_read = True
    elif params.dataset == 'tieredimagenet':
        base_file = 'train'
        novel_file = 'test'
        params.num_classes = 351
    else:
        ValueError('dataset error')

    datamgr = SimpleDataManager(params.data_path, params.image_size, batch_size=params.batch_size, json_read=json_file_read)
    base_loader = datamgr.get_data_loader(base_file, aug=False)
    novel_loader = datamgr.get_data_loader(novel_file, aug=False)
    # loadfile_base = configs.data_dir[params.dataset] + 'base.json'
    # loadfile_novel = configs.data_dir[params.dataset] + 'novel.json'
    # if params.dataset == 'miniImagenet' or params.dataset == 'CUB':
    #     datamgr       = SimpleDataManager(84, batch_size = 256)
    # base_loader = datamgr.get_data_loader(loadfile_base, aug=False)
    # novel_loader      = datamgr.get_data_loader(loadfile_novel, aug = False)

    checkpoint_dir = './checkpoints/%s/%s_%s' % (params.dataset, params.model, params.method)
    # 获取模型文件的路径
    # modelfile   = get_resume_file(checkpoint_dir)

    model = BaselineTrain(params, model_dict[params.model], params.num_classes)
    model = model.cuda()
    model.eval()

    # 加载预训练模型参数
    modelfile = os.path.join(params.save_dir,params.distill_model)
    tmp = torch.load(modelfile)
    state = tmp['state']
    model.load_state_dict(state)
    # if params.model == 'WideResNet28_10':
    #     model = wrn_model.wrn28_10(num_classes=params.num_classes)


    # model = model.cuda()
    # cudnn.benchmark = True

    # 加载模型检查点
    # checkpoint = torch.load(modelfile)
    # state = checkpoint['state']
    # state_keys = list(state.keys())

    # 如果模型保存时使用DataParallel包裹，则需要将包裹的模型取出
    # callwrap = False
    # if 'module' in state_keys[0]:
    #     callwrap = True
    # if callwrap:
    #     model = WrappedModel(model)

    # 更新模型权重
    # model_dict_load = model.state_dict()
    # model_dict_load.update(state)
    # model.load_state_dict(model_dict_load)
    # model.eval()

    # 提取特征
    output_dict_base = extract_feature(base_loader, model, checkpoint_dir, tag='last', set='base')
    print("base set features saved!")
    output_dict_novel=extract_feature(novel_loader, model, checkpoint_dir, tag='last',set='novel')
    print("novel features saved!")
