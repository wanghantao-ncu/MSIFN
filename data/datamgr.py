# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from data.dataset import SetDataset_JSON, SimpleDataset, SetDataset, EpisodicBatchSampler, SimpleDataset_JSON
from abc import abstractmethod

# 图像转换和预处理
class TransformLoader:
    # 类初始化
    def __init__(self, image_size):
        # 图像标准化
        self.normalize_param = dict(mean=[0.472, 0.453, 0.410], std=[0.277, 0.268, 0.285])
        
        self.image_size = image_size
        if image_size == 84:
            self.resize_size = 92
        elif image_size == 128:
            self.resize_size = 140
        elif image_size == 224:
            self.resize_size = 256
    # 返回一个图像预处理的转换函数（aug为True时，进行随机增强，否则不增强）
    def get_composed_transform(self, aug=False):
        if aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalize_param)
            ])
        return transform

# DataManager作为一个基类，为数据加载器提供基础结构
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

# SimpleDataManager管理和提供简化的数据加载器，能够处理图像数据的预处理和加载
class SimpleDataManager(DataManager):
    def __init__(self, data_path, image_size, batch_size, json_read=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.trans_loader = TransformLoader(image_size)
        self.json_read = json_read

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            dataset = SimpleDataset_JSON(self.data_path, data_file, transform)
        else:
            dataset = SimpleDataset(self.data_path, data_file, transform)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

# SetDataManager管理和提供SetData数据集的加载器，能够处理SetData数据集的预处理和加载
class SetDataManager(DataManager):
    def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode, json_read=False,aug_num = 0,args=None):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.data_path = data_path
        self.json_read = json_read
        self.aug_num = aug_num
        self.args = args

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        if self.json_read:
            # print(self.aug_num)
            dataset = SetDataset_JSON(self.data_path, data_file, self.batch_size, transform,aug_num=self.aug_num, args=self.args)
        else:
            dataset = SetDataset(self.data_path, data_file, self.batch_size, transform,aug_num=self.aug_num, args=self.args)
        # sampler为随机way采样器，采样5个way，len(dataset)为类别数量
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        data_loader_params = dict(batch_sampler=sampler, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

# data_loader



