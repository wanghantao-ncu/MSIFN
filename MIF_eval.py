import pickle
import time
import numpy as np
import argparse
import torch
import pprint
import os
import time
from data.datamgr import SetDataManager
from methods.MIF import MIF_Net
from utils.utils import set_seed,load_model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

DATA_DIR = 'data'

# 用了四个线程
torch.set_num_threads(4)

# 打印函数
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

# 解析命令行参数
def parse_option():
    parser = argparse.ArgumentParser('arguments for model pre-train')
    # about dataset and network
    parser.add_argument('--dataset', type=str, default='miniimagenet',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100'])
    parser.add_argument('--data_root', type=str, default=DATA_DIR)
    parser.add_argument('--model', default='ResNet12',choices=['ResNet12', 'ResNet18', 'resnet34', 'conv64'])
    parser.add_argument('--img_size', default=84, type=int, choices=[84,224])
    parser.add_argument('--num_classes', default=64, type=int, choices=[64,351,100])
    parser.add_argument('--train_val_num', default=80, type=int, choices=[80,448,150])
    #parser.add_argument('--transform', default='B')

    # about model :
    parser.add_argument('--drop_gama', default=0.5, type= float)
    parser.add_argument("--beta", default=0.01, type=float)
    parser.add_argument('--drop_rate', default=0.5, type=float)
    parser.add_argument('--reduce_dim', default=128, type=int)

    # about test
    parser.add_argument('--set', type=str, default='test', choices=['val', 'test'], help='the set for validation')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_aug_support_samples',type=int, default=17)
    parser.add_argument('--n_queries', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=2000)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test_batch_size',default=1)
    parser.add_argument('--grid',default=None)

    # setting
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--model_type',default='best',choices=['best','last'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--no_save_model', default=False, action='store_true')
    parser.add_argument('--method',default='stl_deepbdc',choices=['local_proto','good_metric','stl_deepbdc','confusion','WinSA'])
    parser.add_argument('--distill_model', default='tieredimagenet/ResNet12_stl_deepbdc_distill/last_model.tar',type=str,help='about distillation model path')
    parser.add_argument('--penalty_c', default=1.0, type=float)
    parser.add_argument('--test_times', default=1, type=int)

    # confusion representation:
    parser.add_argument('--n_symmetry_aug', default=1, type=int)
    parser.add_argument('--embeding_way', default='BDC', choices=['BDC','GE','protonet','baseline++'])
    parser.add_argument('--LR', default=False,action='store_true')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--fix_seed', default=False, action='store_true')
    parser.add_argument('--local_scale', default=0.2 , type=float)

    parser.add_argument('--measure', default='cosine', choices=['cosine','eudist'])
    parser.add_argument('--dc_k', default=3, type=int)
    parser.add_argument('--semantic_path', default='filelist/miniImageNet/label2vec_glove_miniimagenet.npy',type=str)
    parser.add_argument('--classifier', default='metric', choices=['SFC','metric','LR'])
    parser.add_argument('--step', default='test')
    parser.add_argument('--n_local',type=int, default=3)

    args = parser.parse_args()
    args.n_symmetry_aug = args.n_aug_support_samples

    return args

# 加载预训练模型
def model_load(args,model):
    # method = 'deep_emd' if args.deep_emd else 'local_match'
    method = args.method
    # 构建模型文件保存路径，如checkpoint/miniimagenet_local_proto_resnet12.pth
    save_path = os.path.join(args.save_dir, args.dataset + "_" + method + "_resnet12_"+args.model_type
                                            + ("_"+str(args.model_id) if args.model_id else "") + ".pth")
    # 检查是否提供了蒸馏模型的路径，若有则用save_path保存蒸馏模型路径
    if args.distill_model is not None:
        save_path = os.path.join(args.save_dir, args.distill_model)
    else:
        assert "model load failed! "
    print('teacher model path: ' + save_path)
    # 加载模型权重，存储在state_dict中，通过load_state_dict方法将权重加载到模型中
    state_dict = torch.load(save_path)['model']
    model.load_state_dict(state_dict)
    return model

def remove_feats(base_means,feat_base):
    for i in range(len(feat_base)):
        mean = base_means[i]
        mean = mean.reshape(1, -1)
        matrix_L2_dist = np.linalg.norm(feat_base[i] - mean, axis=-1)
        index = np.argsort(matrix_L2_dist)[:400]
        feat_base[i] = feat_base[i][index]
    return feat_base

def main():
    # 解析命令行参数
    args = parse_option()
    # 模型图像大小调整
    # if args.img_size == 224 and args.transform == 'B':
    #     args.transform = 'B224'
    # 计算数据增强支持样本数，grid数据增强方法,通常设为None
    if args.grid:
        args.n_aug_support_samples = 1
        for i in args.grid:
            args.n_aug_support_samples += i ** 2
        args.n_symmetry_aug = args.n_aug_support_samples

    pprint(args)
    # 环境配置和随机种子设置（这里设置了使用的GPU设备，并检查是否需要固定随机种子，确保实验的可重复性）
    if args.gpu:
        gpu_device = args.gpu
    else:
        gpu_device = "0"
    # CUDA_VISIBLE_DEVICES环境变量控制着CUDA使用哪个GPU进行计算
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    # 设置随机种子来控制随机数生成器的初始状态
    if args.fix_seed:
        set_seed(args.seed)
    # 加载数据集
    json_file_read = False
    if args.dataset == 'cub':
        novel_file = 'novel.json'
        json_file_read = True
    else:
        novel_file = 'test'
    if args.dataset == 'miniimagenet':
        # novel_few_shot_params例子：{'n_way': 5, 'n_support': 1}
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        # **novel_few_shot_params是对字典解包，字典中的每个键值对转换为对应的参数名和参数值，query_num=15, n_episode=2000,aug_num=17
        novel_datamgr = SetDataManager('filelist/miniImageNet', args.img_size, n_query=args.n_queries,
                                       n_episode=args.n_episodes, json_read=json_file_read,aug_num=args.n_aug_support_samples,args=args,
                                       **novel_few_shot_params)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        args.num_classes = 64
    elif args.dataset == 'cub':
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        novel_datamgr = SetDataManager('filelist/CUB',args.img_size, n_query=args.n_queries,
                                       n_episode=args.n_episodes, json_read=json_file_read,aug_num=args.n_aug_support_samples,args=args,
                                       **novel_few_shot_params)
        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        args.num_classes = 100
    elif args.dataset == 'tieredimagenet':
        novel_few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot)
        novel_datamgr = SetDataManager('filelist/tieredImageNet', args.img_size, n_query=args.n_queries,
                                       n_episode=args.n_episodes, json_read=json_file_read,aug_num=args.n_aug_support_samples,args=args,
                                       **novel_few_shot_params)

        novel_loader = novel_datamgr.get_data_loader(novel_file, aug=False)
        args.num_classes = 351
    # 加载基类统计量 checkpoints/miniimagenet/ResNet12_stl_deepbdc/last/base_features.plk
    base_means = []
    base_feat = []
    base_label = []
    # base_cov = []
    base_features_path = './checkpoints/%s/%s_%s/last/base_features.plk' % (args.dataset, args.model, args.method)
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        base_key = list(data.keys())
        for key in data.keys():
            labels = [key] * len(data[key])
            feature = np.array(data[key])
            # mean -miniImageNet(8256,)
            mean = np.mean(feature, axis=0)
            base_label.append(labels)
            base_means.append(mean)
            base_feat.append(feature)
    #base_feat = remove_feats(base_means,base_feat)
    #--emb为array
    emb = np.load(args.semantic_path)
    # 模型初始化和加载
    model = MIF_Net(args,num_classes=args.num_classes).cuda()
    model.eval()
    # 加载预训练模型参数
    model = load_model(model,os.path.join(args.save_dir,args.distill_model))

    # 开始测试
    print("-"*20+"  start meta test...  "+"-"*20)
    acc_sum = 0
    confidence_sum = 0
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    file_path = os.path.join(result_dir, f'results_{timestamp}_{args.dataset}_{args.n_shot}.txt')
    with open(file_path, 'w') as f:
        # log_dir = os.path.join('runs', args.dataset + '_' + str(args.n_shot) + '_' + timestamp)
        # writer = SummaryWriter(log_dir=log_dir)
        for t in range(args.test_times):
            with torch.no_grad():
                tic = time.time()
                # mean平均准确率，confidence置信度
                mean, confidence = model.meta_test_loop(novel_loader,data,base_key,base_means,base_feat,base_label,emb)
                acc_sum += mean
                confidence_sum += confidence
                print()
                print("Time {} :meta_val acc: {:.2f} +- {:.2f}   elapse: {:.2f} min".format(t,mean * 100, confidence * 100,
                                                                                (time.time() - tic) / 60))
                f.write("Time {} :meta_val acc: {:.2f} +- {:.2f}   elapse: {:.2f} min\n".format(t, mean * 100, confidence * 100, (time.time() - tic) / 60))
        # 输出结果
        print("{} times \t acc: {:.2f} +- {:.2f}".format(args.test_times, acc_sum/args.test_times * 100, confidence_sum/args.test_times * 100, ))
        f.write("{} times \t acc: {:.2f} +- {:.2f}\n".format(args.test_times, acc_sum / args.test_times * 100, confidence_sum / args.test_times * 100))

if __name__ == '__main__':
    main()