import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

import cv2
from layers.module.reverse_grad import ReverseGrad
from models.resnet import resnet50, embed_net, convDiscrimination, Discrimination, resnet34, gem
from utils.calc_acc import calc_acc

from layers import TripletLoss,RerankLoss
from layers import CenterTripletLoss
from layers import CenterLoss
from PIL import Image
import os
from layers import cbam
from layers import NonLocalBlockND


def rns(self, w, v, rns_indices):
    bs, hn, dl, _ = w.shape
    rns_indices = rns_indices.unsqueeze(1).repeat(1, hn, 1, 1)
    mask = torch.zeros_like(w).scatter_(3, rns_indices, torch.ones_like(rns_indices, dtype=w.dtype))
    mask = mask * mask.transpose(2, 3)
    if 'cuda' in str(w.device):
        mask = mask.cuda()
    else:
        mask = mask.cpu()
    if self.training:
        w = w * mask + -1e9 * (1 - mask)
        w = F.softmax(w, dim=3)
        a_v = torch.matmul(w, v)
    else:
        w = (w * mask).reshape(bs * hn, dl, dl).to_sparse()
        w = torch.sparse.softmax(w, 2)
        v = v.reshape(bs * hn, dl, -1)
        a_v = torch.bmm(w, v).reshape(bs, hn, dl, -1)
    return a_v

def spearman_loss(dist_matrix, rerank_matrix):
    # 获取排序的索引
    sorted_idx_dist = torch.argsort(dist_matrix, dim=1)
    sorted_idx_rerank = torch.argsort(rerank_matrix, dim=1)
    # 计算 Spearman's rank correlation
    # 注意：这只是一个简化的版本，可能需要进一步优化以提高效率
    rank_corr = 0
    n = dist_matrix.size(1)
    for i in range(dist_matrix.size(0)):
        diff = sorted_idx_dist[i] - sorted_idx_rerank[i]
        rank_corr += 1 - (6 * torch.sum(diff * diff) / (n * (n**2 - 1)))
    # 取平均值
    rank_corr /= dist_matrix.size(0)
    # 返回损失
    return 1 - rank_corr
def pairwise_dist(x, y):
    # Compute pairwise distance of vectors
    xx = (x**2).sum(dim=1, keepdim=True)
    yy = (y**2).sum(dim=1, keepdim=True).t()
    dist = xx + yy - 2.0 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist

def kl_soft_dist(feat_st):
    n_st = feat_st.size(0)
    # dist_st = torch.cdist(feat_st, feat_st).clamp(min=1e-6)
    dist_st = pairwise_dist(feat_st, feat_st)

    #mask_st = labels.expand(n_st, n_st).eq(labels.expand(n_st, n_st).t())
    mask_st_1 = torch.ones(n_st, n_st, dtype=bool)
    for i in range(n_st):  # 将同一类样本中自己与自己的距离舍弃
        mask_st_1[i, i] = 0
    '''mask_st_2 = []
    for i in range(n_st):  
        # f_d_ap.append(dist_f[i][mask[i]].sum()-dist_f_diag[i])
        mask_st_2.append(mask_st[i][mask_st_1[i]])  # 后面是布尔，前面不是布尔
    mask_st_2 = torch.stack(mask_st_2)'''
    dist_st_2 = []
    for i in range(n_st):
        # f_d_ap.append(dist_f[i][mask[i]].sum()-dist_f_diag[i])
        dist_st_2.append(dist_st[i][mask_st_1[i]])
    dist_st_2 = torch.stack(dist_st_2)
    # dist_st_2 = F.softmax(-(dist_st_2 - 1), 1)
    return dist_st_2
def Bg_kl(logits1, logits2):####输入:(60,206),(60,206)
    ####双KL
    KL = nn.KLDivLoss(reduction='batchmean')
    kl_loss_12 = KL(F.log_softmax(logits1, 1), F.softmax(logits2, 1))
    kl_loss_21 = KL(F.log_softmax(logits2, 1), F.softmax(logits1, 1))
    bg_loss_kl = kl_loss_12 + kl_loss_21
    return kl_loss_12, bg_loss_kl
def Sm_kl(logits1, logits2, labels):
    #######双KL
    KL = nn.KLDivLoss(reduction='batchmean')
    #n_kl = logits1.size(0)  # 120
    #v_logits = logits[sub == 0]
    #i_logits = logits[sub == 1]
    #mask_kl = labels.expand(n_kl, n_kl).eq(labels.expand(n_kl, n_kl).t())
    #m_kl = int((mask_kl[0] == True).sum() / 2)  # 5
    # m_kl = (labels == labels[0]).sum() // 2
    m_kl = torch.div((labels == labels[0]).sum(), 2, rounding_mode='floor')
    v_logits_s = logits1.split(m_kl, 0)
    i_logits_s = logits2.split(m_kl, 0)
    sm_v_logits = torch.cat(v_logits_s, 1)  # .t()  # 5,206*12->206*12,5
    sm_i_logits = torch.cat(i_logits_s, 1)  # .t()
    sm_kl_loss_vi = KL(F.log_softmax(sm_v_logits, 1), F.softmax(sm_i_logits, 1))
    sm_kl_loss_iv = KL(F.log_softmax(sm_i_logits, 1), F.softmax(sm_v_logits, 1))
    sm_kl_loss = sm_kl_loss_vi + sm_kl_loss_iv
    return sm_kl_loss_vi, sm_kl_loss

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PatchPrompter(nn.Module):
    def __init__(self, image_size):
        super(PatchPrompter, self).__init__()
        image_height, image_width = image_size

        self.prompt = nn.Parameter(torch.randn([1, 3, image_height, image_width]))

    def forward(self, x):
        # image_np = self.prompt.detach().squeeze().permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
        #
        # # 将 NumPy 数组转换为 PIL 图像
        # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围
        #
        # # 指定保存路径
        # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
        # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
        #
        # # 保存图像
        # save_path = os.path.join(save_dir, "ppp.png")
        # image_pil.save(save_path)  # 你可以更改文件名和格式
        # import pdb
        # pdb.set_trace()

        prompt = torch.cat(x.size(0) * [self.prompt])
        return x + prompt

class PadPrompter(nn.Module):
    def __init__(self, pad_size, image_size):
        super(PadPrompter, self).__init__()
        #pad_size = args.prompt_size
        image_height, image_width = image_size

        self.base_height = image_height - pad_size * 2
        self.base_width = image_width - pad_size * 2

        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_width]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_width]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, self.base_height, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, self.base_height, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_height, self.base_width).cuda()  # .to(device) #.cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat([prompt] * x.size(0))  # torch.cat(x.size(0) * [prompt])
        return x + prompt
def generate_adversarial_labels(labels, num_classes):
    batch_size = labels.size(0)
    adversarial_labels = torch.zeros(batch_size, num_classes)

    for i in range(batch_size):
        # 设置每个类别的概率为 1/395
        adversarial_labels[i] = torch.ones(num_classes) / num_classes

    return adversarial_labels

class Baseline(nn.Module):
    def __init__(self, num_classes=None, drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False,decompose=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning
        self.decompose = decompose
        #self.backbone = embed_net(drop_last_stride=drop_last_stride, modality_attention=modality_attention,decompose=decompose)
        self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
        self.resnet34 = resnet34(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)

        self.base_dim = 2048
        self.dim = 0
        self.part_num = kwargs.get('num_parts', 0)


        if pattern_attention:
            self.base_dim = 2048
            self.dim = 2048
            self.part_num = kwargs.get('num_parts', 6)
            self.spatial_attention = nn.Conv2d(self.base_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
            torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
            self.activation = nn.Sigmoid()
            self.weight_sep = kwargs.get('weight_sep', 0.1)

        if mutual_learning:
            self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

            self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.visible_classifier_.weight.requires_grad_(False)
            self.visible_classifier_.weight.data = self.visible_classifier.weight.data

            self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.infrared_classifier_.weight.requires_grad_(False)
            self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

            self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')
            self.weight_sid = kwargs.get('weight_sid', 0.5)
            self.weight_KL = kwargs.get('weight_KL', 2.0)
            self.update_rate = kwargs.get('update_rate', 0.2)
            self.update_rate_ = self.update_rate

        print("output feat length:{}".format(self.base_dim + self.dim * self.part_num))
        self.bn_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck.bias, 0) 
        self.bn_neck.bias.requires_grad_(False)
        self.bn_neck_sp = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        nn.init.constant_(self.bn_neck_sp.bias, 0)
        self.bn_neck_sp.bias.requires_grad_(False)

        if kwargs.get('eval', False):
            return

        self.classification = kwargs.get('classification', False)
        self.triplet = kwargs.get('triplet', False)
        self.center_cluster = kwargs.get('center_cluster', False)
        self.center_loss = kwargs.get('center', False)
        self.margin = kwargs.get('margin', 0.3)
        self.bg_kl = kwargs.get('bg_kl', False)
        self.sm_kl = kwargs.get('sm_kl', False)
        self.distalign = kwargs.get('distalign', False)
        self.prompt = kwargs.get('prompt', False)
        self.pad = kwargs.get('padding_size', 20)
        self.image_size = kwargs.get('image_size', (384,128))

        if self.prompt:
            # self.PrompterV = PadPrompter(pad_size = self.pad, image_size = self.image_size)
            # self.PrompterI = PadPrompter(pad_size=self.pad, image_size=self.image_size)
            self.PrompterV = PatchPrompter(image_size=self.image_size)
            self.PrompterI = PatchPrompter(image_size=self.image_size)
            self.grl0 = ReverseGrad(alpha=0.0001)
            self.grl1 = ReverseGrad(alpha=100000)
            #self.grl = ReverseGrad()
            self.D_k = Discrimination(512)
        if self.decompose:
            self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_sp = nn.Linear(self.base_dim, num_classes, bias=False)
            #self.classifier_sp_new = nn.Linear(self.base_dim, 9, bias=False)
            self.D_special = Discrimination()
            self.D_shared_pseu = Discrimination(2048)
            #self.grl = ReverseGrad()
        else:
            self.classifier_new = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_V = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
            self.classifier_I = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

            #self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        if self.mutual_learning or self.classification:
            self.id_loss = nn.CrossEntropyLoss(ignore_index=-1)
            self.last_acc = 0
        if self.triplet:
            self.triplet_loss = TripletLoss(margin=self.margin)
            self.rerank_loss = RerankLoss(margin=1.7)
        if self.center_cluster:
            k_size = kwargs.get('k_size', 8)
            self.center_cluster_loss = CenterTripletLoss(k_size=k_size, margin=self.margin)
        if self.center_loss:
            self.center_loss = CenterLoss(num_classes, self.base_dim + self.dim * self.part_num)
    def get_causality_loss(self, x_IN_entropy, x_useful_entropy): #, x_useless_entropy
        self.ranking_loss = torch.nn.SoftMarginLoss()

        y = torch.ones_like(x_IN_entropy)
        # import pdb
        # pdb.set_trace()
        return self.ranking_loss(x_IN_entropy - x_useful_entropy, y) #+ self.ranking_loss(x_useless_entropy - x_IN_entropy, y)

    def get_entropy(self, p_softmax):
        # exploit ENTropy minimization (ENT) to help DA,
        mask = p_softmax.ge(0.000001)
        mask_out = torch.masked_select(p_softmax, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return (entropy / float(p_softmax.size(0)))

    def forward(self, inputs, labels=None, **kwargs):
        loss_reg = 0
        loss_center = 0
        modality_logits = None
        modality_feat = None

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6) #0 visible
        #epoch = kwargs.get('epoch')

        # CNN
        #sh_feat, sp_feat, sh_pl, sp_f_pl, sp_pl, IN_pl, x2, x3 = self.backbone(inputs)

        if self.training:
            inputs[sub == 0] = self.PrompterV(inputs[sub == 0])


            inputs[sub == 1] = self.PrompterI(inputs[sub == 1])

            image_np = inputs[20].detach().permute(1, 2, 0).cpu().numpy()  # 调整维度顺序


            # 将 NumPy 数组转换为 PIL 图像
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围

            # 指定保存路径
            save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
            os.makedirs(save_dir, exist_ok=True)  #确保保存文件夹存在，如果不存在则创建

            # 保存图像
            save_path = os.path.join(save_dir, "saved_image.jpg")
            image_pil.save(save_path)  # 你可以更改文件名和格式

            import pdb
            pdb.set_trace()

            image_np = inputs[21].detach().permute(1, 2, 0).cpu().numpy()  # 调整维度顺序

            # 将 NumPy 数组转换为 PIL 图像
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围

            # 指定保存路径
            save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
            os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建

            # 保存图像
            save_path = os.path.join(save_dir, "saved_image.jpg")
            image_pil.save(save_path)  # 你可以更改文件名和格式

            import pdb
            pdb.set_trace()

            # inputsV = self.PrompterV(inputs[sub == 0])
            # inputsI = self.PrompterI(inputs[sub == 1])
            sh_pl = self.backbone(inputs)
            sh_pl = gem(sh_pl).squeeze()  # Gem池化
            sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化

            # 按交替顺序合并矩阵
            inputs_grl = inputs
            inputs_grl = self.grl1(inputs_grl)

            # inputs_grl[sub == 0] = self.grl1(inputs_grl[sub == 0])
            # inputs_grl[sub == 1] = self.grl1(inputs_grl[sub == 1])
            merged_pl = self.backbone(inputs_grl)
            merged_pl = gem(merged_pl).squeeze()  # Gem池化
            merged_pl = merged_pl.view(merged_pl.size(0), -1)  # Gem池化
        else:
            if (sub == 0).all():
                # image_np = inputs[2].permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
                #
                # # 将 NumPy 数组转换为 PIL 图像
                # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围
                #
                # # 指定保存路径
                # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
                # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
                #
                # # 保存图像
                # save_path = os.path.join(save_dir, "saved_image.jpg")
                # image_pil.save(save_path)  # 你可以更改文件名和格式
                #
                # import pdb
                # pdb.set_trace()
                #inputs = inputs
                inputs = self.PrompterV(inputs)

                # image_np = inputs[2].permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
                #
                # # 将 NumPy 数组转换为 PIL 图像
                # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围
                #
                # # 指定保存路径
                # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
                # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
                #
                # # 保存图像
                # save_path = os.path.join(save_dir, "saved_image.jpg")
                # image_pil.save(save_path)  # 你可以更改文件名和格式
                #
                import pdb
                pdb.set_trace()


            elif (sub == 1).all():
                # image_np = inputs[2].permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
                #
                # # 将 NumPy 数组转换为 PIL 图像
                # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围
                #
                # # 指定保存路径
                # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
                # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
                #
                # # 保存图像
                # save_path = os.path.join(save_dir, "ori.jpg")
                # image_pil.save(save_path)  # 你可以更改文件名和格式


                inputs = self.PrompterI(inputs)

                #inputs = inputs

                # image_np = inputs[2].permute(1, 2, 0).cpu().numpy()  # 调整维度顺序
                #
                # # 将 NumPy 数组转换为 PIL 图像
                # image_pil = Image.fromarray((image_np * 255).astype(np.uint8))  # 这里假设输入的张量是标准化的，所以需要还原到 0-255 范围
                #
                # # 指定保存路径
                # save_dir = "/home/zhang/E/RKJ/MAPnet/new-classifier-prompt/prompt"
                # os.makedirs(save_dir, exist_ok=True)  # 确保保存文件夹存在，如果不存在则创建
                #
                # # 保存图像
                # save_path = os.path.join(save_dir, "saved_image.jpg")
                # image_pil.save(save_path)  # 你可以更改文件名和格式
                #
                # import pdb
                # pdb.set_trace()
            else:
                raise ValueError("Invalid sub values in test mode")

            x_sh = self.backbone(inputs)
            sh_pl = gem(x_sh).squeeze()  # Gem池化
            sh_pl = sh_pl.view(sh_pl.size(0), -1) # Gem池化


        # x_sh = self.backbone(inputs)
        # sh_pl = gem(x_sh).squeeze()  # Gem池化
        # sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化

        # if self.training:
        #     x_sh = self.backbone(inputs)
        #     x_k_sh = self.resnet34(self.grl(inputs))
        #
        #     sh_pl = gem(x_sh).squeeze()  # Gem池化
        #     sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化
        #
        #     sh_k_pl = gem(x_k_sh).squeeze()  # Gem池化
        #     sh_k_pl = sh_k_pl.view(sh_k_pl.size(0), -1)  # Gem池化
        #     # import pdb
        #     # pdb.set_trace()
        # else:
        #     x_sh = self.backbone(inputs)
        #     sh_pl = gem(x_sh).squeeze()  # Gem池化
        #     sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化




        if self.pattern_attention:
            global_feat = sh_feat
            b, c, w, h = global_feat.shape
            masks = global_feat
            masks = self.spatial_attention(masks)
            masks = self.activation(masks)

            feats = []
            for i in range(self.part_num):
                mask = masks[:, i:i+1, :, :]
                feat = mask * global_feat

                feat = F.avg_pool2d(feat, feat.size()[2:])
                feat = feat.view(feat.size(0), -1)

                feats.append(feat)

            global_feat = F.avg_pool2d(global_feat, global_feat.size()[2:])
            global_feat = global_feat.view(global_feat.size(0), -1)

            feats.append(global_feat)
            feats = torch.cat(feats, 1)

            if self.training:
                masks = masks.view(b, self.part_num, w*h)
                loss_reg = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_reg = torch.triu(loss_reg, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)

        else:
            feats = sh_pl

        if not self.training:
            feats = self.bn_neck(feats)
            return feats
        else:
            #return self.train_forward(feats, sp_pl, labels,
                                      #loss_reg, sub, IN_pl, sh_feat,sp_f_pl, **kwargs)
            return self.train_forward(feats, merged_pl, labels,
                                      loss_reg, sub, **kwargs)



    def train_forward(self, feat,merged_pl, labels,
                      loss_reg, sub, **kwargs):
        epoch = kwargs.get('epoch')
        metric = {}
        if self.pattern_attention and loss_reg != 0 :
            loss = loss_reg.float() * self.weight_sep
            metric.update({'p-reg': loss_reg.data})
        else:
            loss = 0

        if self.triplet:
            # F_feat = F.normalize(feat, dim=1)
            # F_sp_pl = F.normalize(sp_pl, dim=1)


            triplet_loss, dist, sh_ap, sh_an = self.triplet_loss(feat.float(), labels)

            #triplet_loss_im, _, sp_ap, sp_an = self.triplet_loss(sp_pl.float(), labels)

            # triplet_loss_IN, sp_IN_ap, sp_IN_an = self.triplet_loss(IN_pl.float(), labels)
            # loss_d_recover_p = self.get_causality_loss(sp_IN_ap, sp_ap)
            # loss_d_recover_n = self.get_causality_loss(sp_an, sp_IN_an)
            # loss_d_recover = loss_d_recover_p + loss_d_recover_n


            trip_loss = triplet_loss #+ triplet_loss_im  # + triplet_loss_IN  #+ triplet_loss_sp

            loss += trip_loss  # + loss_d_recover #* 0.1 #loss_causality#+ ft_dist
            metric.update({'tri': trip_loss.data})

        if False:
            rkl_loss, dsit_rk, _, _ = self.rerank_loss(feat.float(), labels)

            #loss_align = spearman_loss(dist, dsit_rk)
            loss += rkl_loss  #20 * loss_align +
            metric.update({'rekl': rkl_loss.data})
            ##metric.update({'align': loss_align.data})


        bb = 90
        if self.distalign:

            sf_sp_dist_v = kl_soft_dist(sp_pl[sub == 0])
            sf_sp_dist_i = kl_soft_dist(sp_pl[sub == 1])
            sf_sh_dist_v = kl_soft_dist(feat[sub == 0])
            sf_sh_dist_i = kl_soft_dist(feat[sub == 1])

            _, kl_inter_v = Bg_kl(sf_sh_dist_v, sf_sp_dist_v)
            _, kl_inter_i = Bg_kl(sf_sh_dist_i, sf_sp_dist_i)


            _, kl_intra = Bg_kl(sf_sh_dist_v, sf_sh_dist_i)


            if feat.size(0) == bb:
                soft_dt = kl_intra + (kl_inter_v + kl_inter_i) * 0.7

            else:
                soft_dt = (kl_intra + kl_inter_v + kl_inter_i) * 0.1  # 95

            loss += soft_dt
            metric.update({'soft_dt': soft_dt.data})

        if self.center_loss:
            center_loss = self.center_loss(feat.float(), labels)
            loss += center_loss
            metric.update({'cen': center_loss.data})

        if self.center_cluster:
            center_cluster_loss, _, _ = self.center_cluster_loss(feat.float(), labels)
            loss += center_cluster_loss
            metric.update({'cc': center_cluster_loss.data})

        feat = self.bn_neck(feat)
        #sp_pl = self.bn_neck_sp(sp_pl)

        if self.decompose:
            sub_nb = sub + 0  ##模态标签

            logits_sp = self.classifier_sp(sp_pl)  # self.bn_neck_un(sp_pl)
            loss_id_sp = self.id_loss(logits_sp.float(), labels)



            # sp_f_logits = self.D_special(sp_f_pl)  ####鉴别器梯度反转
            # # sp_logits = self.D_special(sp_pl)
            # unad_loss_f = self.id_loss(sp_f_logits.float(), sub_nb)

            sp_logits = self.D_special(sp_pl)
            unad_loss_b = self.id_loss(sp_logits.float(), sub_nb)
            unad_loss = unad_loss_b

            # 翻转标签
            # sub_flp = sub_nb.flip(0)

            pseu_sh_logits = self.D_shared_pseu(feat)
            p_sub = sub_nb.chunk(2)[0].repeat_interleave(2)
            pp_sub = torch.roll(p_sub, -1)
            pseu_loss = self.id_loss(pseu_sh_logits.float(), pp_sub)

            loss += loss_id_sp + unad_loss + pseu_loss

            metric.update({'unad': unad_loss.data})
            metric.update({'id_pl': loss_id_sp.data})

            metric.update({'pse': pseu_loss.data})

        if self.classification:
            #############
            # sub_nb = sub + 0
            # k_logits = self.D_k(sh_k_pl)
            # ad_loss = self.id_loss(k_logits.float(), sub_nb)
            # #loss += ad_loss
            #############

            logits = self.classifier_new(feat)
            one_hot_labels = F.one_hot(labels, 395)

            # 取反
            flipped_one_hot_labels = (1 - one_hot_labels)/394

            # adversarial_labels = generate_adversarial_labels(labels[sub == 1], 395)
            # adversarial_labelsV = torch.randint(0, 395, size=(feat[sub==1].size(0),))
            # adversarial_labelsI = torch.randint(0, 395, size=(feat[sub == 1].size(0),))
            # import pdb
            # pdb.set_trace()


            logits_VV = self.classifier_V(feat[sub==0])
            logits_II = self.classifier_I(feat[sub==1])
            # import pdb
            # pdb.set_trace()

            logits_VI = self.grl0(self.classifier_I(merged_pl[sub == 0]))
            logits_IV = self.grl0(self.classifier_V(merged_pl[sub == 1]))
            # logits_VI = self.classifier_I(self.grl(feat[sub == 0]))
            # logits_IV = self.classifier_V(self.grl(feat[sub == 1]))

            cls_loss_VV = self.id_loss(logits_VV.float(), labels[sub == 0])
            cls_loss_II = self.id_loss(logits_II.float(), labels[sub == 1])

            cls_loss_VI = self.id_loss(logits_VI.float(), labels[sub == 0])
            cls_loss_IV = self.id_loss(logits_IV.float(), labels[sub == 1])
            # cls_loss_VI = self.id_loss(logits_VI.float(), flipped_one_hot_labels[sub == 0].float())
            # cls_loss_IV = self.id_loss(logits_IV.float(), flipped_one_hot_labels[sub == 1].float())

            cls_ho = (cls_loss_VV + cls_loss_II)/2
            cls_he = (cls_loss_VI + cls_loss_IV)/2

            acc_ho = (calc_acc(logits_VV.data, labels[sub == 0]) + calc_acc(logits_II.data, labels[sub == 1]))/2
            acc_he = (calc_acc(logits_VI.data, labels[sub == 0]) + calc_acc(logits_IV.data, labels[sub == 1]))/2

            loss += cls_ho + 0.1*cls_he

            metric.update({'acc_ho': acc_ho, 'ce_ho': cls_ho.data})
            metric.update({'acc_he': acc_he, 'ce_he': cls_he.data})
            #metric.update({'ad': ad_loss.data})





            #unique_labels, labels = torch.unique(labels, return_inverse=True)

            # for name, param in model.named_parameters():
            #     print(name)
            # import pdb
            # pdb.set_trace()
            if self.bg_kl:#60:60
                _, inter_bg_v = Bg_kl(logits[sub == 0], logits_sp[sub == 0])
                _, inter_bg_i = Bg_kl(logits[sub == 1], logits_sp[sub == 1])

                _, intra_bg = Bg_kl(logits[sub == 0], logits[sub == 1])
                #intra_bg = JS_loss(logits[sub == 0], logits[sub == 1])

                if feat.size(0) == bb:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.7  # intra_bg + (inter_bg_v + inter_bg_i) * 0.7
                    # bg_loss = intra_bg + inter_bg * 0.3

                else:
                    bg_loss = intra_bg + (inter_bg_v + inter_bg_i) * 0.3
                loss += bg_loss #* 0.3
                metric.update({'bg_kl': bg_loss.data})

            if self.sm_kl:#(5,206)*12
                _, inter_Sm_v = Sm_kl(logits[sub == 0], logits_sp[sub == 0], labels)
                _, inter_Sm_i = Sm_kl(logits[sub == 1], logits_sp[sub == 1], labels)
                inter_Sm = inter_Sm_v + inter_Sm_i
                _, intra_Sm = Sm_kl(logits[sub == 0], logits[sub == 1], labels)

                if feat.size(0) == bb:
                    sm_kl_loss = intra_Sm * 0.5 + inter_Sm * 0.7

                else:
                    sm_kl_loss = intra_Sm + inter_Sm * 0.3
                loss += sm_kl_loss
                metric.update({'sm_kl': sm_kl_loss.data})

            cls_loss = self.id_loss(logits.float(), labels)
            loss += cls_loss
            acc = calc_acc(logits.data, labels)
            acc_diff = acc - self.last_acc
            self.last_acc = acc
            # import pdb
            # pdb.set_trace()
            metric.update({'acc': acc, 'ce': cls_loss.data})


        if self.mutual_learning:
            # cam_ids = kwargs.get('cam_ids')
            # sub = (cam_ids == 3) + (cam_ids == 6)
            
            logits_v = self.visible_classifier(feat[sub == 0])
            v_cls_loss = self.id_loss(logits_v.float(), labels[sub == 0])
            loss += v_cls_loss * self.weight_sid
            logits_i = self.infrared_classifier(feat[sub == 1])
            i_cls_loss = self.id_loss(logits_i.float(), labels[sub == 1])
            loss += i_cls_loss * self.weight_sid

            logits_m = torch.cat([logits_v, logits_i], 0).float()
            with torch.no_grad():
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.infrared_classifier.weight.data * self.update_rate
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.visible_classifier.weight.data * self.update_rate

                logits_v_ = self.infrared_classifier_(feat[sub == 0])
                logits_i_ = self.visible_classifier_(feat[sub == 1])

                logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
            logits_m = F.softmax(logits_m, 1)
            logits_m_ = F.log_softmax(logits_m_, 1)
            mod_loss = self.KLDivLoss(logits_m_, logits_m) 

            loss += mod_loss * self.weight_KL + (v_cls_loss + i_cls_loss) * self.weight_sid
            metric.update({'ce-v': v_cls_loss.data})
            metric.update({'ce-i': i_cls_loss.data})
            metric.update({'KL': mod_loss.data})

        return loss, metric #, acc #_diff
