import os
import sys
import time

import pickle
import torch.cuda
import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from methods.bdc_module import BDC
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import pulp as lp
# from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

sys.path.append("..")
import scipy
from scipy.stats import t
import network.resnet as resnet
from utils.loss import *
from sklearn.linear_model import LogisticRegression as LR
from utils.loss import DistillKL
from utils.utils import *
import math
from torch.nn.utils.weight_norm import WeightNorm
from methods.ProtoFusion import LocalFeatureExtractionModule

import warnings
warnings.filterwarnings("ignore")
# 计算输入数据的均值和置信区间
def mean_confidence_interval(data, confidence=0.95,multi = 1):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m * multi, h * multi

# 对输入张量进行L2归一化处理
def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

# 从给定范围内随机抽取样本索引
def random_sample(linspace, max_idx, num_sample=5):
    sample_idx = np.random.choice(range(linspace), num_sample)
    sample_idx += np.sort(random.sample(list(range(0, max_idx, linspace)),num_sample))
    return sample_idx

# 将三维张量转换为一维向量
def Triuvec(x,no_diag = False):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y

# 将五维张量转换为较低维度的表示，提取上三角部分信息
def Triumap(x,no_diag = False):

    batchSize, dim, dim, h, w = x.shape
    r = x.reshape(batchSize, dim * dim, h, w)
    I = torch.ones(dim, dim).triu()
    if no_diag:
        I -= torch.eye(dim,dim)
    I  = I.reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    # y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index, :, :].squeeze()
    return y

# 在给定的三维张量中提取每个二维矩阵的上三角部分元素，并将这些元素重塑为一维向量
def Diagvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.eye(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = r[:, index].squeeze()
    return y

# 无偏置项的全连接层
class FClayer(nn.Module):

    def __init__(self, z_out, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.z_out = z_out
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.z_out, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        # self.fc1_b = nn.Parameter(torch.zeros(self.z_out))
        # self.vars.append(self.fc1_b)
    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        #fc1_b = the_vars[1]
        #net = F.linear(input_x, fc1_w, fc1_b)
        net = F.linear(input_x, fc1_w)
        return net

    def parameters(self):
        return self.vars
    
class MIF_Net(nn.Module):
    def __init__(self,params,num_classes = 5,):
        super(MIF_Net, self).__init__()

        self.params = params

        if params.model == 'ResNet12':
            # 这里的num_classes表示该模型输出维度是64
            self.feature = resnet.ResNet12(avg_pool=True,num_classes=64)
            resnet_layer_dim = [64, 160, 320, 640]
        elif params.model == 'ResNet18':
            self.feature = resnet.ResNet18()
            resnet_layer_dim = [64, 128, 256, 512]

        self.resnet_layer_dim = resnet_layer_dim
        self.reduce_dim = params.reduce_dim
        # 将已定义好的特征提取网络的特征维度赋值 feat_dim=[640,10,10]
        self.feat_dim = self.feature.feat_dim
        # 计算当前特征的最终维度 dim=8256
        self.dim = int(self.reduce_dim * (self.reduce_dim+1)/2)
        # 对特征提取网络最后一层操作，将其输出维度调整为reduce_dim(128)
        if resnet_layer_dim[-1] != self.reduce_dim:

            self.Conv = nn.Sequential(
                nn.Conv2d(resnet_layer_dim[-1], self.reduce_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.reduce_dim),
                nn.ReLU(inplace=True)
            )
            self._init_weight(self.Conv.modules())

        self.dcov = BDC(is_vec=True, input_dim=[self.reduce_dim,self.feature.feat_dim[1],self.feature.feat_dim[2]], dimension_reduction=self.reduce_dim)
        # 将(N, C, H, W)->(N, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if resnet_layer_dim[-1] != self.reduce_dim:
            self.dcov.conv_dr_block = self.Conv

        self.n_shot = params.n_shot
        self.n_way = params.n_way
        self.n_queries = params.n_queries
        self.transform_aug = params.n_aug_support_samples
        self.n_local = params.n_local

        self.BaseFeatureExtractionModule = LocalFeatureExtractionModule(8256, 129, 1)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.BaseFeatureExtractionModule = self.BaseFeatureExtractionModule.cuda()

        self.mseloss = torch.nn.MSELoss()
        if params.dataset =='cub':
            self.Nete = nn.Linear(8256, 312)
            self.Netd = nn.Linear(312, 8256)
        else:
            self.Nete = nn.Linear(8256, 300)
            self.Netd = nn.Linear(300, 8256)

        self.encoder_train_count = 0
        self.proto_fusion_train_count = 0
        self.sfc_count = 0
        self.transform_localtrain_count = 0
        self.transform_basetrain_count = 0

    # 初始化网络权重，对卷积层和批归一化层进行初始化
    def _init_weight(self,modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    # 对于输入张量进行均值归一化
    def normalize(self,x):
        x = (x - torch.mean(x, dim=1).unsqueeze(1))
        return x
    # 从输入图像中提取特征表示
    def forward_feature(self, x):
        feat_map = self.feature(x, )
        if self.resnet_layer_dim[-1] != self.reduce_dim:
            feat_map = self.Conv(feat_map)
        out = feat_map
        return out

    # 元学习测试阶段
    def meta_test_loop(self,test_loader,data,base_key,base_means,base_feat,base_label,emb):
        acc = []
        for i, (x, y_label) in enumerate(test_loader):
            # 仅在第一个 episode 可视化
            visualize_flag = (i == self.params.n_episodes - 1)
            self.params.n_aug_support_samples = self.transform_aug
            tic = time.time()
            # x = [n_way,(n_shot+n_queries),transform_aug,c,h,w]-miniImageNet(5,20,17,3,84,84) y_label(n_way,n_shot+n_queries)
            x = x.contiguous().view(self.n_way, (self.n_shot + self.params.n_queries), *x.size()[2:])
            # 选择每个类别的前n_shot个样本作为支持集， support_xs = [n_way * n_shot * n_aug_support_samples,c,h,w]-miniImageNet(425,3,84,84)
            support_xs = x[:, :self.n_shot].contiguous().view(
                self.n_way * self.n_shot * self.params.n_aug_support_samples, *x.size()[3:]).cuda()
            # 选择每个类别的后n_queries个样本作为查询集，query_xs = [n_way * n_queries * n_symmetry_aug,c,h,w]-miniImageNet(1275,3,84,84)
            query_xs = x[:, self.n_shot:, 0:self.params.n_symmetry_aug].contiguous().view(
                self.n_way * self.params.n_queries * self.params.n_symmetry_aug, *x.size()[3:]).cuda()
            # 创建支持集标签张量，数量为n_way * n_shot * n_aug_support_samples   -miniImageNet(1,425)
            support_y = torch.from_numpy(np.repeat(range(self.params.n_way),self.n_shot*self.params.n_aug_support_samples)).unsqueeze(0)

            split_size = 128
            if support_xs.shape[0] >= split_size:
                feat_sup_ = []
                for j in range(math.ceil(support_xs.shape[0]/split_size)):
                    fest_sup_item =self.forward_feature(support_xs[j*split_size:min((j+1)*split_size,support_xs.shape[0]),],)
                    feat_sup_.append(fest_sup_item if len(fest_sup_item.shape)>=1 else fest_sup_item.unsqueeze(0))
                # 提取到支持集的批次特征-miniImageNet(425,128,10,10)
                feat_sup = torch.cat(feat_sup_,dim=0)
            else:
                feat_sup = self.forward_feature(support_xs)
            if query_xs.shape[0] >= split_size:
                feat_qry_ = []
                for j in range(math.ceil(query_xs.shape[0]/split_size)):
                    feat_qry_item = self.forward_feature(
                        query_xs[j * split_size:min((j + 1) * split_size, query_xs.shape[0]), ],)
                    feat_qry_.append(feat_qry_item if len(feat_qry_item.shape) > 1 else feat_qry_item.unsqueeze(0))
                # 提取到的查询集批次特征-miniImageNet(1275,128,10,10)
                feat_qry = torch.cat(feat_qry_,dim=0)
            else:
                feat_qry = self.forward_feature(query_xs,)


            if self.params.classifier == 'metric':
                with torch.enable_grad():
                    pred = self.softmax(feat_sup, support_y, y_label, feat_qry, data, base_key, base_means,base_feat,base_label,emb,visualize=visualize_flag, episode_idx=i)
                query_ys = np.repeat(range(self.n_way), self.params.n_queries)
                acc_epo = np.mean(pred.cpu().numpy() == query_ys)
            else:
                with torch.enable_grad():
                    # feat_sup支持集特征，support_y支持集标签，feat_qry查询集特征，pred为预测结果(n_way,n_queries,n_way)
                    pred = self.softmax(feat_sup, support_y, y_label, feat_qry, data, base_key, base_means,base_feat,base_label,emb,visualize=visualize_flag, episode_idx=i)
                    _,pred = torch.max(pred,dim=-1)
                query_ys = np.repeat(range(self.n_way), self.params.n_queries)
                pred = pred.view(-1)
                acc_epo = np.mean(pred.cpu().numpy() == query_ys)
            acc.append(acc_epo)
            print("\repisode {} acc: {:.2f} | avg_acc: {:.2f} +- {:.2f}, elapse : {:.2f}".format(i, acc_epo * 100,
                                                                                                 *mean_confidence_interval(
                                                                                                     acc, multi=100), (
                                                                                                             time.time() - tic) / 60),
                  end='')
        # writer.close()
        return mean_confidence_interval(acc)
    #--根据语义选择最近基类
    def get_cos_similar_matrix_loop(self, v1, v2):
        simmat = np.zeros((v1.shape[0], v2.shape[0]))
        for k in range(v1.shape[0]):
            for i in range(v2.shape[0]):
                num = np.dot(v1[k], v2[i])
                denom = np.linalg.norm(v1[k]) * np.linalg.norm(v2[i])
                #num1 = np.dot(proto_moving_array[k], base_means[i])
                #denom1 = np.linalg.norm(proto_moving_array[k]) * np.linalg.norm(base_means[i])
                res = num / denom
                #res1 = num1 / denom1
                simmat[k, i] = res
        return 0.5 + 0.5 * simmat
    
    def get_l2_distance_matrix_loop(self, v1, v2):
        distmat = np.zeros((v1.shape[0], v2.shape[0]))
        for k in range(v1.shape[0]):
            for i in range(v2.shape[0]):
                # 计算L2距离（欧几里得距离）
                distance = np.linalg.norm(v1[k] - v2[i])
                distmat[k, i] = distance
        return distmat
    # 路径规划选择唯一的基类
    def route_plan_J(self, Dij):
        NN, BB = Dij.shape
        model = lp.LpProblem(name='plan_0_1', sense=lp.LpMaximize)
        x = [[lp.LpVariable("x_{},{}".format(i, j), cat="Binary") for j in range(BB)] for i in range(NN)]
        # objective
        objective = 0
        for i in range(NN):
            for j in range(BB):
                objective = objective + Dij[i, j] * x[i][j]
        model += objective
        # constraints
        for i in range(NN):
            in_degree = 0
            for j in range(BB):
                in_degree = in_degree + x[i][j]
            model += in_degree == 1
        for j in range(BB):
            out_degree = 0
            for i in range(NN):
                out_degree = out_degree + x[i][j]
            model += out_degree <= 1
        model.solve(lp.apis.PULP_CBC_CMD(msg=False))

        W = np.zeros((NN, BB))
        for v in model.variables():
            idex = [int(s) for s in v.name.split('_')[1].split(',')]
            W[idex[0], idex[1]] = v.varValue
        return W
    
    def base_data_novel_choose(self, feat_g, base_feat, base_label, maxid, k):
        feat_b = []
        abs_lb = []
        n_way, n_shot, n_dim = feat_g.shape
        for labell in maxid:
            for i in range(len(base_label)):
                if base_label[i][0] == labell:
                    id = i
            feat_b.append(base_feat[id])
            abs_lb.extend([labell]*k)

        # 存储每组的gathered特征
        gathered_feat_b_list = []

        for i in range(n_way):
            # 将当前组的feat_b转换为tensor
            feat_b_i = torch.from_numpy(feat_b[i]).to(device=0)  # shape: (L_i, n_dim)
            
            # 获取对应组的feat_g
            feat_g_i = feat_g[i]  # shape: (n_shot, n_dim)
            
            # 计算距离矩阵: (n_shot, L_i)
            matrix_L2_dist = torch.linalg.norm(feat_g_i.unsqueeze(1) - feat_b_i.unsqueeze(0), dim=-1)
            
            # top-k操作
            index = torch.topk(matrix_L2_dist, k, dim=-1, largest=False, sorted=True).indices  # shape: (n_shot, k)
            
            # gather操作
            # 扩展feat_b_i以匹配gather的维度要求
            feat_b_expanded = feat_b_i.unsqueeze(0).expand(n_shot, -1, n_dim)  # shape: (n_shot, L_i, n_dim)
            
            # 扩展index以匹配feat_b_expanded的维度
            index_expanded = index.unsqueeze(-1).expand(n_shot, k, n_dim)  # shape: (n_shot, k, n_dim)
            
            # gather操作
            gathered_feat_b_i = torch.gather(feat_b_expanded, dim=1, index=index_expanded)  # shape: (n_shot, k, n_dim)
            
            gathered_feat_b_list.append(gathered_feat_b_i)

        # 如果你需要将所有结果合并成一个张量
        # 注意：由于每组的k值相同，但L_i可能不同，这里可以安全堆叠
        gathered_feat_b = torch.stack(gathered_feat_b_list)  # shape: (n_way, n_shot, k, n_dim)
        
        return gathered_feat_b, abs_lb
    
    def feat_local_base_fusion(self, feat_novel_b, feat_g):
        n_way, n_shot, n_dim = feat_g.shape
        matrix_L2_dist = torch.linalg.norm(feat_g.unsqueeze(-2) - feat_novel_b, dim=-1)
        Weight = torch.div(1, 1 + matrix_L2_dist)
        Weight_gathered_feat_novel_b = torch.matmul(feat_novel_b.permute(0, 1, 3, 2), Weight.unsqueeze(-1)).reshape(n_way, n_shot, n_dim)
        # Weight_gathered_feat_novel_b = Weight_gathered_feat_novel_b.reshape(n_way*n_shot, n_dim)
        # feat_g_sl = feat_g_sl.reshape(n_way*n_shot, n_dim)
        # Weight = Weight.reshape(n_way*n_shot, -1)
        learned_mean = torch.div((Weight_gathered_feat_novel_b + feat_g),
                                torch.sum(Weight.unsqueeze(-1), dim=2) + 1)
        return learned_mean
    
    def encoder_train(self,feat_s,sem):

            optimizer1 = torch.optim.Adam([
                {'params': self.Nete.parameters()},
                {'params': self.Netd.parameters()}
            ], lr=1e-4, weight_decay=1e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.2)

            self.Nete.train()
            self.Netd.train()
            for i in range(40):
                self.encoder_train_count = self.encoder_train_count + 1
                laten = self.Nete(feat_s)
                losslatent = self.mseloss(laten, sem)
                rec = self.Netd(laten)
                lossrec = self.mseloss(rec, feat_s)
                loss = losslatent + 0.5 * lossrec

                optimizer1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer1.step()
                lr_scheduler.step()

                # writer.add_scalar('Loss/losslatent', losslatent.item(), self.encoder_train_count)
                # writer.add_scalar('Loss/lossrec', lossrec.item(), self.encoder_train_count)
                # writer.add_scalar('Loss/encoder_trainloss', loss.item(), self.encoder_train_count)
            self.Nete.eval()
            self.Netd.eval()

    # 绘制对比 t-SNE
    def plot_comparison_tsne(self, feat_pre, feat_post, feat_query, labels_sup, labels_qry, n_way, n_shot, dataset, episode_idx):
        """
        绘制对比 t-SNE: 左图(Pre-Fusion) vs 右图(Post-Fusion)
        feat_pre:  融合前的支持集特征 (n_support, dim)
        feat_post: 融合后的支持集特征 (n_support, dim)
        feat_query: 查询集特征 (n_query, dim)
        """
        print(f"Generating comparison t-SNE for episode {episode_idx}...")
        
        # 初始化 TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=0, learning_rate='auto')
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        titles = [f't-SNE on {dataset} before fusion', f't-SNE on {dataset} after fusion']
        
        # 颜色表
        colors = plt.cm.rainbow(np.linspace(0, 1, n_way))

        # --- 循环处理两组数据：0=Pre, 1=Post ---
        for idx, feat_sup in enumerate([feat_pre, feat_post]):
            ax = axes[idx]
            
            # 1. 拼接 Support 和 Query 以统一 t-SNE 空间
            # 注意：Query 特征在对比中保持不变，作为参照系
            X_combined = np.concatenate([feat_sup, feat_query], axis=0)
            X_embedded = tsne.fit_transform(X_combined)
            
            num_sup = feat_sup.shape[0]
            X_sup_2d = X_embedded[:num_sup]
            X_qry_2d = X_embedded[num_sup:]
            
            # 2. 绘图
            for k in range(n_way):
                # 绘制 Support Set (圆点)
                idx_s = np.where(labels_sup == k)[0]
                # 注意：删除了 label 参数
                ax.scatter(X_sup_2d[idx_s, 0], X_sup_2d[idx_s, 1], c=[colors[k]], marker='o', 
                        s=80, edgecolors='k', alpha=0.7)
                
                # 绘制 Query Set (叉号)
                idx_q = np.where(labels_qry == k)[0]
                # 注意：删除了 label 参数
                ax.scatter(X_qry_2d[idx_q, 0], X_qry_2d[idx_q, 1], c=[colors[k]], marker='x', 
                        s=80, linewidths=2, alpha=0.9)

            ax.set_title(titles[idx], fontsize=16)
            # 去掉坐标轴刻度，让图更干净（可选，不需要可以注释掉这两行）
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        save_path = f'tsne_{dataset}_{n_way}way_{n_shot}shot_ep{episode_idx}.png'
        plt.savefig(save_path)
        print(f"Saved comparison plot to {save_path}")
        plt.close()
    # new selective local fusion :
    def softmax(self,support_z,support_ys, y_label, query_z, data, base_key, base_means,base_feat,base_label,emb,visualize=False, episode_idx=0):
        support_ys = support_ys.cuda()

        if self.params.classifier == 'metric':
            if self.params.embeding_way in ['BDC']:
                # -miniImageNet(425,8256) (n_way * n_shot * n_aug_support_samples,feature_dim) 
                support_z = self.dcov(support_z)
                # -miniImageNet(1275,8256) (n_way * n_queries * n_aug_support_samples,feature_dim)
                query_z = self.dcov(query_z)
            
            # support_ys = [n_way * n_shot, n_aug_support_samples, -1],-1根据其他维度自动计算 -miniImageNet(25,17,1)
            support_ys = support_ys.view(self.n_way * self.n_shot, self.params.n_aug_support_samples, -1)
            # global_ys = [n_way * n_shot, -1]是每个类别的第一个样本的标签集合 -miniImageNet(25,1)
            global_ys = support_ys[:, 0, :]
            
            # support_z -miniImageNet(5,5,17,8256) 17是全局特征和局部特征拼接而的
            support_z = support_z.reshape(self.n_way,self.n_shot,self.params.n_aug_support_samples,-1)
            #support_z_up = self.Distribution_fitting_with_DDWM(data, base_key, support_z, base_means, self.params.dc_k, self.params.dc_delta, self.params.dc_gamma)
            #fused_support_z = 0.5*support_z + 0.5*support_z_up
            # query_z -miniImageNet(5,15,17,8256)
            query_z = query_z.reshape(self.n_way,self.params.n_queries,self.params.n_aug_support_samples,-1)

            # 查询样本全局特征 在增强样本中选择每个类的第一个样本的特征表示 (n_way,n_queries,feature_dim)-miniImageNet(5,15,8256)
            feat_q = query_z[:,:,0]
            # 查询样本局部特征 从查询特征中剔除了第一个增强样本，提取了其余查询样本的特征表示 (n_way,n_queries,idx_walk,feature_dim)--miniImageNet(5,15,16,8256)
            feat_ql = query_z[:,:,1:]
            # feat_g（支持样本全局特征）在增强样本中选择每个类的第一个样本的特征表示 (n_way,n_shot,feature_dim)-miniImageNet(5,5,8256)
            feat_g = support_z[:,:,0]
            # feat_sl（裁剪后图像局部特征）从支持特征中剔除了第一个增强样本，提取了其余支持样本的特征表示 (n_way,n_shot,idx_walk,feature_dim)--miniImageNet(5,5,16,8256)
            feat_sl = support_z[:,:,1:]
            # w_local: n * k * n * m
            # 重塑global_ys张量(n_way,n_shot,1)(5,5,1)
            global_ys = global_ys.view(self.n_way,self.n_shot,-1)
            feat_sl = feat_sl.detach()
            feat_g = feat_g.detach()

            if self.params.dataset != 'cub':
                y_temp = torch.add(y_label[:, 0], self.params.train_val_num)
            else:
                y_temp = y_label[:, 0]
            emb_novel = emb[y_temp]

            if self.params.dataset == 'cub':
                emb_base = emb[::2]  # 选择 emb 中所有偶数索引的部分
            else:
                emb_base = emb[:self.params.num_classes] 

            #--(5,64)
            emb_novel_base = self.get_cos_similar_matrix_loop(emb_novel, emb_base)
            maxid = np.where(self.route_plan_J(emb_novel_base) == 1)[1]
            if self.params.dataset == 'cub':
                maxid = maxid * 2
                feat_novel_b, abs_lb = self.base_data_novel_choose(feat_g, base_feat, base_label, maxid, self.params.dc_k)
                sem_b = emb[abs_lb]
                sem_b1 = emb[maxid]
            else:
                maxid = maxid
                feat_novel_b, abs_lb = self.base_data_novel_choose(feat_g, base_feat, base_label, maxid, self.params.dc_k)
                sem_b = emb_base[abs_lb]
                sem_b1 = emb_base[maxid]
            
            feat_ns = support_z[:,:,0].view(-1, support_z.size(-1))
            # feat_ns = support_x.view(-1, support_x.size(-1))
            sem_ns = emb_novel.repeat(self.params.n_shot, 0)
            sem_n1 = emb_novel
            sem_b = sem_b.reshape(self.n_way, self.params.dc_k, -1)

            if torch.cuda.is_available():
                sem_b = torch.tensor(sem_b).type(torch.cuda.FloatTensor)
                sem_b1 = torch.tensor(sem_b1).type(torch.cuda.FloatTensor)
                sem_ns = torch.tensor(sem_ns).type(torch.cuda.FloatTensor)
                sem_n1 = torch.tensor(sem_n1).type(torch.cuda.FloatTensor)
            sem_b = sem_b.unsqueeze(1).repeat(self.n_shot, 1, 1, 1)
            sem_b = sem_b.reshape(self.n_way*self.n_shot, self.params.dc_k, -1)

            self.encoder_train(feat_ns,sem_ns)

            feat_sl = feat_sl.reshape(self.n_way,self.n_shot, self.params.n_aug_support_samples-1,-1)
            # feat_sl = F.normalize(feat_sl, dim=-1)
            feat_sl_sem = self.Nete(feat_sl)

            # cos
            matrix_cosine_dist = torch.cosine_similarity(sem_ns.reshape(self.n_way,self.n_shot, -1).unsqueeze(2), feat_sl_sem, dim=-1)
            matrix_dist = 1 - matrix_cosine_dist
            index = torch.topk(matrix_dist, self.n_local, dim=-1, largest=False, sorted=True).indices
            feat_sl_gathered = torch.gather(feat_sl, dim=-2, index=index.unsqueeze(-1).expand(self.n_way, self.n_shot, self.n_local, self.dim))
            
            # Weight = torch.div((2 - matrix_dist), 2)

            # gather_weight = torch.gather(Weight, dim=-1, index=index).unsqueeze(-1).reshape(self.n_way,self.n_shot, self.n_local, 1)

            # Weight_feat_sl = torch.matmul(feat_sl_gathered.permute(0, 1, 3, 2), gather_weight).reshape(self.n_way, self.n_shot, self.dim)
            # feat_g_sl_cos = torch.div((Weight_feat_sl + feat_g.reshape(self.n_way, self.n_shot, self.dim)),
            #                             torch.sum(gather_weight, dim=2) + 1)

            # #这里用feat_g_sl_cos进行选择基类样本和feat_g融合会不会更好
            # feat_g_b_cos = self.feat_local_base_fusion(feat_novel_b, feat_g)
            # #这里是手动加权，能不能自适应加权，如feat_g_sl_cos和feat_g_b_cos转换为sem信息，和类真实语义信息的近似度作为权重
            # feat_g_sl_b = 0.9*feat_g_sl_cos + 0.1*feat_g_b_cos

            # proto_moving = torch.mean(feat_g_sl_b, dim=1)
            proto_moving = torch.mean(feat_g, dim=1)
            if visualize:
                feat_pre_fusion_cpu = feat_g.detach().cpu().numpy().reshape(self.params.n_way * self.params.n_shot, -1)
            feat_g = feat_g.view(self.params.n_way*self.params.n_shot, -1)
            feat_sl_gathered = feat_sl_gathered.view(self.n_way*self.n_shot, self.n_local, self.dim)
            feat_novel_b = feat_novel_b.view(self.n_way*self.n_shot, self.params.dc_k, self.dim)
            proto = proto_moving.repeat(1, self.n_shot).view(proto_moving.size(0) * self.n_shot, proto_moving.size(1))
            sem_ns = sem_ns.reshape(self.n_way*self.n_shot, -1)

            # 基类信息融合
            optimizer_base = torch.optim.Adam([
                {'params': self.BaseFeatureExtractionModule.parameters(), 'lr': 1e-3}])
            lr_scheduler_base = torch.optim.lr_scheduler.StepLR(optimizer_base, step_size=250, gamma=0.1)

            self.BaseFeatureExtractionModule.train()
            for i in range(250):
                self.transform_basetrain_count = self.transform_basetrain_count + 1
                optimizer_base.zero_grad()

                feat_g_b = self.BaseFeatureExtractionModule(feat_g, feat_sl_gathered, feat_novel_b)

                # 以proto为标准
                total_loss_base = self.mseloss(feat_g_b, proto)

                # 先用encode转为语义，然后和标准语义比较
                sem_g_b = self.Nete(feat_g_b)
                loss_sem = self.mseloss(sem_g_b, sem_ns)

                loss = loss_sem + total_loss_base
                # loss = total_loss_base
                # loss = loss_sem

                loss.backward(retain_graph=True)
                optimizer_base.step()
                lr_scheduler_base.step()
                # writer.add_scalar('Loss/baseloss', loss.item(), self.transform_basetrain_count)

            self.BaseFeatureExtractionModule.eval()

            feat_g_b = self.BaseFeatureExtractionModule(feat_g, feat_sl_gathered, feat_novel_b)

            feat_q = feat_q.view(self.params.n_way*self.params.n_queries,-1)
            feat_g = feat_g.view(self.params.n_way*self.params.n_shot,-1)

            # # === 插入点 2: 执行可视化 ===
            # if visualize:
            #         # 融合后的特征也立即转到 CPU
            #         z_post_cpu = feat_g_b.detach().cpu().numpy() 
            #         z_qry_cpu = feat_q.detach().cpu().numpy()
                    
            #         # 标签生成 (直接用 numpy，不要用 torch)
            #         y_sup_cpu = np.repeat(range(self.params.n_way), self.n_shot)
            #         y_qry_cpu = np.repeat(range(self.params.n_way), self.params.n_queries)
                    
            #         # 调用绘图 (确保传入的全是 CPU numpy 数据)
            #         self.plot_comparison_tsne(feat_pre_fusion_cpu, z_post_cpu, z_qry_cpu, y_sup_cpu, y_qry_cpu, self.params.n_way, self.params.n_shot, self.params.dataset, episode_idx)
                    
            #         # 绘图完手动清理一下
            #         del feat_pre_fusion_cpu, z_post_cpu, z_qry_cpu
            #         torch.cuda.empty_cache()
            
            out = self.LR(feat_g_b, feat_q)

        else:
            raise ValueError(f"Unsupported method: {self.params.classifier}")

        return out
    # 逻辑回归进行分类
    def LR(self,support_z,query_z):

        clf = LR(penalty='l2',
                 random_state=0,
                 C=self.params.penalty_c,
                 solver='lbfgs',
                 max_iter=1000,
                 multi_class='multinomial')

        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        spt_normalized = support_z.div(spt_norm  + 1e-6)

        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        qry_normalized = query_z.div(qry_norm + 1e-6)

        z_support = spt_normalized.detach().cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        y_support = np.repeat(range(self.params.n_way), self.n_shot)

        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))

    def SVM(self,support_z,query_z):

        clf = SVC(C=10, gamma='auto', kernel='linear', probability=True)

        spt_norm = torch.norm(support_z, p=2, dim=1).unsqueeze(1).expand_as(support_z)
        spt_normalized = support_z.div(spt_norm  + 1e-6)

        qry_norm = torch.norm(query_z, p=2, dim=1).unsqueeze(1).expand_as(query_z)
        qry_normalized = query_z.div(qry_norm + 1e-6)

        z_support = spt_normalized.detach().cpu().numpy()
        z_query = qry_normalized.detach().cpu().numpy()

        y_support = np.repeat(range(self.params.n_way), self.n_shot)

        clf.fit(z_support, y_support)

        return torch.from_numpy(clf.predict(z_query))