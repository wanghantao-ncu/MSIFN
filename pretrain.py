import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
import argparse

from data.datamgr import SimpleDataManager, SetDataManager
from methods.template import BaselineTrain
from utils.utils import *

def train(params, base_loader, val_loader, model, stop_epoch):

    trlog = {}
    trlog['args'] = vars(params)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    trlog['max_acc_epoch'] = 0

    if params.method in ['stl_deepbdc', 'meta_deepbdc']:
        # freeze the temperature parameter
        bas_params = filter(lambda p: id(p) != id(model.dcov.temperature), model.parameters())
        optimizer = torch.optim.SGD([
            {'params': bas_params},
            {'params': model.dcov.temperature, 'lr': params.t_lr}], lr=params.lr, weight_decay=params.wd, nesterov=True, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, weight_decay=params.wd, nesterov=True, momentum=0.9)
    # 创建一个学习率调度器，该调度器将在指定的训练轮次（milestones）自动降低学习率，以帮助模型在训练早期快速收敛，并在训练后期进行精细调整。
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=params.milestones, gamma=params.gamma)


    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    count = 0
    for epoch in range(0, stop_epoch):
        if epoch%2==0:
            params.classification_epoch=0
            params.gaussian_epoch=1
        else:
            params.classification_epoch = 1
            params.gaussian_epoch = 0

        start = time.time()
        model.train()
        trainObj, top1 = model.train_loop(epoch, base_loader, optimizer,count)

        if count == params.classification_epoch + params.gaussian_epoch - 1:
            count = 0
        else:
            count = count + 1

        if params.dataset == 'tiered_imagenet':
            if epoch >= params.milestones[1]:
                model.eval()
                if params.val in ['meta']:
                    if params.val == 'meta':
                        valObj, acc = model.meta_test_loop(val_loader)
                    trlog['val_loss'].append(valObj)
                    trlog['val_acc'].append(acc)
                    if acc > trlog['max_acc']:
                        print("best model! save...")
                        trlog['max_acc'] = acc
                        trlog['max_acc_epoch'] = epoch
                        outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                        torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                    print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
                    print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))
        else:
            model.eval()
            if params.val in ['meta']:
                if params.val == 'meta':
                    valObj, acc = model.meta_test_loop(val_loader)
                trlog['val_loss'].append(valObj)
                trlog['val_acc'].append(acc)
                if acc > trlog['max_acc']:
                    print("best model! save...")
                    trlog['max_acc'] = acc
                    trlog['max_acc_epoch'] = epoch
                    outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                    torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
                print("val loss is {:.2f}, val acc is {:.2f}".format(valObj, acc))
                print("model best acc is {:.2f}, best acc epoch is {}".format(trlog['max_acc'], trlog['max_acc_epoch']))
        if epoch % params.save_freq == 0:
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
        
        if epoch == stop_epoch - 1:
            outfile = os.path.join(params.checkpoint_dir, 'last_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        trlog['train_loss'].append(trainObj)
        trlog['train_acc'].append(top1)

        torch.save(trlog, os.path.join(params.checkpoint_dir, 'trlog'))

        lr_scheduler.step()

        print("This epoch use %.2f minutes" % ((time.time() - start) / 60))
        print("train loss is {:.2f}, train acc is {:.2f}".format(trainObj, top1))    

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_size', default=84, type=int, choices=[84, 224], help='input image size, 84 for miniimagenet and tieredimagenet, 224 for cub')
    parser.add_argument('--batch_size', default=64, type=int, help='pre-training batch size')
    parser.add_argument('--lr', type=float, default=5e-2, help='initial learning rate of the backbone')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--t_lr', type=float, default=1e-3, help='initial learning rate uesd for the temperature of bdc module')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--milestones', nargs='+', type=int, default=[100, 150], help='milestones for MultiStepLR')
    parser.add_argument('--epoch', default=170, type=int, help='stopping epoch')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--step', default='pretrain')

    parser.add_argument('--dataset', default='miniImagenet', choices=['miniImagenet','tieredImagenet','CUB'])
    parser.add_argument('--data_path', default='filelist/miniImageNet', type=str, help='dataset path')

    parser.add_argument('--model', default='ResNet12', choices=['ResNet12', 'ResNet18','WRN28'])
    parser.add_argument('--method', default='meta_deepbdc', choices=['meta_deepbdc', 'stl_deepbdc', 'protonet', 'good_embed'])
    
    parser.add_argument('--val', default='meta', choices=['meta', 'last'], help='validation method')
    parser.add_argument('--val_n_episode', default=1000, type=int, help='number of episodes in meta validation')
    parser.add_argument('--val_n_way', default=5, type=int, help='number of  classes to classify in meta validation')
    parser.add_argument('--n_shot', default=1, type=int, help='number of labeled data in each class, same as n_support during meta validation')
    parser.add_argument('--n_query', default=15, type=int, help='number of unlabeled data in each class during meta validation')

    parser.add_argument('--extra_dir', default='', help='recording additional information')
    parser.add_argument('--num_classes', default=64, type=int, help='total number of classes in training')
    parser.add_argument('--save_freq', default=100, type=int, help='the frequency of saving model .pth file')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    parser.add_argument('--reduce_dim', default=128, type=int, help='the output dimensions of BDC dimensionality reduction layer')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate for pretrain and distillation')
    parser.add_argument('--classification_epoch', default=0, type=int, help='')
    parser.add_argument('--grid',default=None)
    parser.add_argument('--n_aug_support_samples',type=int, default=17)
    parser.add_argument('--local_scale', default=0.2 , type=float)

    params = parser.parse_args()

    num_gpu = set_gpu(params)
    set_seed(params.seed)


    if params.val == 'last':
        val_file = None
    elif params.val == 'meta':
        val_file = 'val'

    json_file_read = False
    if params.dataset == 'miniImagenet':
        base_file = 'train'
        params.num_classes = 64
    elif params.dataset == 'CUB':
        base_file = 'base.json'
        val_file =  'val.json'
        json_file_read = True
        params.num_classes = 200
    elif params.dataset == 'tieredImagenet':
        base_file = 'train'
        params.num_classes = 351
    else:
        ValueError('dataset error')

    base_datamgr = SimpleDataManager(params.data_path, params.img_size, batch_size=params.batch_size, json_read=json_file_read)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    if params.val == 'meta':
        test_few_shot_params = dict(n_way=params.val_n_way, n_support=params.n_shot)
        val_datamgr = SetDataManager(params.data_path, params.img_size, n_query=params.n_query, n_episode=params.val_n_episode, json_read=json_file_read, aug_num=params.n_aug_support_samples,args=params,**test_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    else:
        val_loader = None

    model = BaselineTrain(params, model_dict[params.model], params.num_classes)

    model = model.cuda()

    params.checkpoint_dir = './checkpoint/%s/%s_%s' % (params.dataset, params.model, params.method)
    params.checkpoint_dir += '_pretrain'
    params.checkpoint_dir += params.extra_dir

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    print(params)
    model = train(params, base_loader, val_loader, model, params.epoch)
