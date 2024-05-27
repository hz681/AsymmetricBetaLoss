import torch
from torch import nn
from torchvision import models
# import torch.nn.functional as F
import numpy as np
from scipy.special import comb


# class MLL_Model(nn.Module):
#     def __init__(self, n_classes):
#         super(MLL_Model, self).__init__()
#         self.n_classes = n_classes
#         self.resnet = models.resnet50(pretrained=True)
#         self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes)

#     def forward(self, x):
#         return self.resnet(x)


# class EDL_Model(nn.Module):
#     def __init__(self, n_classes):
#         super(EDL_Model, self).__init__()
#         self.n_classes = n_classes
#         self.resnet = models.resnet50(pretrained=True)
#         self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes*2)

#     def forward(self, x):
#         output = self.resnet(x)
#         B_alpha, B_beta = torch.split(output, self.n_classes, 1)
#         B_alpha = torch.exp(B_alpha) + 1
#         B_beta = torch.exp(B_beta) + 1
#         return B_alpha, B_beta



# class clssimp(nn.Module):
#     def __init__(self, ch=2880, num_classes=20):

#         super(clssimp, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.way1 = nn.Sequential(
#             nn.Linear(ch, 1000, bias=True),
#             nn.BatchNorm1d(1000),
#             nn.ReLU(inplace=True),
#         )

#         self.cls= nn.Linear(1000, num_classes, bias=True)

#     def forward(self, x):
#         # bp()
#         x = self.pool(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.way1(x)
#         logits = self.cls(x)
#         return logits

#     def intermediate_forward(self, x):
#         x = self.pool(x)
#         x = x.reshape(x.size(0), -1)
#         x = self.way1(x)
#         return x



# class EDL_DenseNet_Model(nn.Module):
#     def __init__(self, n_classes):
#         super(EDL_DenseNet_Model, self).__init__()
#         self.n_classes = n_classes
#         orig_densenet = models.densenet121(pretrained=True)
#         features = list(orig_densenet.features)
#         self.model = nn.Sequential(*features, nn.ReLU(inplace=True))
#         self.clsfier = clssimp(1024, self.n_classes*2)

#     def forward(self, x):
#         x = self.model(x)
#         output = self.clsfier(x)
#         B_alpha, B_beta = torch.split(output, self.n_classes, 1)
#         B_alpha = torch.exp(B_alpha) + 1
#         B_beta = torch.exp(B_beta) + 1
#         return B_alpha, B_beta


# class EDL_TH_Model(nn.Module):
#     def __init__(self, n_classes):
#         super(EDL_TH_Model, self).__init__()
#         self.n_classes = n_classes
#         self.resnet = models.resnet50(pretrained=True)
#         self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, n_classes*3)
#         # self.elu = nn.ELU()

#     def forward(self, x):
#         output = self.resnet(x)
#         logit, B_alpha, B_beta = torch.split(output, self.n_classes, 1)
#         # B_alpha = self.elu(B_alpha) + 2
#         # B_beta = self.elu(B_beta) + 2
#         B_alpha = torch.exp(B_alpha) + 1
#         B_beta = torch.exp(B_beta) + 1
#         return logit, B_alpha, B_beta


# class EDL_Model_Test(nn.Module):
#     def __init__(self, n_classes):
#         super(EDL_Model_Test, self).__init__()
#         self.n_classes = n_classes
#         resnet = models.resnet50(pretrained=True)
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.fc1 = torch.nn.Linear(2048, n_classes)
#         self.mlp_proj = nn.Sequential(
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=False),
#             nn.Linear(2048, 512)
#         )
#         self.fc2 = torch.nn.Linear(512, n_classes*2)

#     def forward(self, x):
#         feat = self.resnet(x).squeeze()
#         logits = self.fc1(feat)
#         feat_proj = self.mlp_proj(feat)
#         out = self.fc2(feat_proj)
#         B_alpha, B_beta = torch.split(out, self.n_classes, 1)
#         B_alpha = torch.exp(B_alpha) + 1
#         B_beta = torch.exp(B_beta) + 1
#         return logits, B_alpha, B_beta


class EDLModel(nn.Module):
    def __init__(self, n_classes):
        super(EDLModel, self).__init__()
        self.n_classes = n_classes
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.proj1 = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 512)
        )
        self.proj2 = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 512)
        )
        self.fc1 = torch.nn.Linear(512, n_classes*2)
        self.fc2 = torch.nn.Linear(512, n_classes*2)

    def forward(self, x):
        feat = self.resnet(x).squeeze(2).squeeze(2)
        feat_proj1 = self.proj1(feat)
        feat_proj2 = self.proj2(feat)
        out1 = self.fc1(feat_proj1)
        out2 = self.fc2(feat_proj2)
        out1 = torch.exp(out1) + 1
        out2 = torch.exp(out2) + 1
        alpha_in, beta_in = torch.split(out1, self.n_classes, 1)
        alpha_out, beta_out = torch.split(out2, self.n_classes, 1)
        return alpha_in, beta_in, alpha_out, beta_out




# class AsymmetricLoss_EDL(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True, reduction='none'):
#         super(AsymmetricLoss_EDL, self).__init__()

#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, B_alpha, B_beta, y):
#         """"
#         Parameters
#         ----------
#         x: input logits
#         y: targets (multi-label binarized vector)
#         """

#         # Calculating Probabilities
#         # x_sigmoid = torch.sigmoid(x)
#         xs_pos = B_alpha / (B_alpha + B_beta)
#         xs_neg = 1 - xs_pos

#         # Asymmetric Clipping
#         # if self.clip is not None and self.clip > 0:
#         #     xs_neg = (xs_neg + self.clip).clamp(max=1)
#         #     xs_pos_m = 1 - xs_neg

#         # Basic CE calculation
#         # los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
#         # los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
#         los_pos = y * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha))
#         # B_alpha_m = B_beta * (xs_pos_m + self.clip) / (xs_neg - self.clip)
#         los_neg = (1 - y) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta))
#         loss = los_pos + los_neg

#         # Asymmetric Focusing
#         if self.gamma_neg > 0 or self.gamma_pos > 0:
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(False)
#             pt0 = xs_pos * y
#             pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
#             pt = pt0 + pt1
#             one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
#             one_sided_w = torch.pow(1 - pt, one_sided_gamma)
#             if self.disable_torch_grad_focal_loss:
#                 torch.set_grad_enabled(True)
#             loss *= one_sided_w

#         return loss


# class EDL_Loss_Test(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.2, eps=1e-8, disable_torch_grad_focal_loss=True, reduction='none'):
#         super(EDL_Loss_Test, self).__init__()

#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.clip = clip
#         self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self, B_alpha, B_beta, y):

#         los_pos = y * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha))
#         los_neg = (1 - y) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta))

#         torch.set_grad_enabled(False)
#         pt = B_alpha / (B_alpha + B_beta)
#         pm = (pt - self.clip).clamp(min=0)
#         w_pos = torch.pow(1 - pt, self.gamma_pos)
#         w_neg = torch.pow(pm, self.gamma_neg)
#         torch.set_grad_enabled(True)
#         loss = w_pos * los_pos + w_neg * los_neg
#         return loss


# class EDL_Loss_Test2(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2):
#         super(EDL_Loss_Test2, self).__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip

#     def forward(self, B_alpha, B_beta, y):

#         loss = (y * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha))
#                 + (1 - y) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))

#         torch.set_grad_enabled(False)
#         pt = B_alpha / (B_alpha + B_beta)
#         pm = (pt - self.clip).clamp(min=0)
#         w_pos = torch.pow(1 - pt, self.gamma_pos)
#         w_neg = torch.pow(pm, self.gamma_neg)
#         w = y * w_pos + (1 - y) * w_neg
#         torch.set_grad_enabled(True)
#         return w * loss


# class EDL_Loss_Test3(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2):
#         super(EDL_Loss_Test3, self).__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip

#     def forward(self, B_alpha, B_beta, y):

#         # loss = (y * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha))
#         #         + (1 - y) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))
        
#         Lp = torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)
#         Ln = torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)

#         torch.set_grad_enabled(False)
#         pt = B_alpha / (B_alpha + B_beta)
#         pm = (pt - self.clip).clamp(min=0)
#         w_pos = torch.pow(1 - pt, self.gamma_pos)
#         w_neg = torch.pow(pm, self.gamma_neg)
#         torch.set_grad_enabled(True)
#         return w_pos * y * y * Lp + w_neg * (1 - y) * (1-y) * Ln


# class EDL_Loss_Test4(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2):
#         super(EDL_Loss_Test4, self).__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip

#     def forward(self, B_alpha, B_beta, y):
        
#         Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
#         Ln = torch.digamma(B_alpha + B_beta + self.gamma_neg) - torch.digamma(B_beta)

#         torch.set_grad_enabled(False)
#         w_pos = torch.ones_like(B_alpha, dtype=float)
#         w_neg = torch.ones_like(B_beta, dtype=float)
#         # w_pos = w_neg = 1
#         for i in range(self.gamma_pos):
#             w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
#         for i in range(self.gamma_neg):
#             w_neg *= (B_alpha + i) / (B_alpha + B_beta + i)
#         pe = B_alpha / (B_alpha + B_beta)
#         # torch.where(pt <= self.clip)
#         w_neg[torch.where(pe <= self.clip)] = 0
#         # print('pos', w_pos[0])
#         # print('neg', w_neg[0])
#         torch.set_grad_enabled(True)
#         return w_pos * y * Lp + w_neg * (1-y) * Ln


# class EDL_Loss_Test5(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2):
#         super(EDL_Loss_Test5, self).__init__()
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg
#         self.clip = clip

#     def forward(self, B_alpha, B_beta, y):
        
#         Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
#         Ln = torch.digamma(B_alpha + B_beta + self.gamma_neg) - torch.digamma(B_beta)

#         torch.set_grad_enabled(False)
#         w_pos = torch.ones_like(B_alpha, dtype=float)
#         w_neg = torch.zeros_like(B_beta, dtype=float)
#         p_pow = torch.ones_like(B_alpha, dtype=float)
#         c = torch.tensor(self.clip)
#         # print(c.shape, c.dtype)
#         m = torch.min(c, B_alpha / (B_alpha + B_beta))
#         # w_pos = w_neg = 1
#         for i in range(self.gamma_pos):
#             w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
#         lmd = self.gamma_neg
#         for i in range(lmd+1):
#             w_neg += comb(lmd, i) * torch.pow(-m, lmd-i) * p_pow
#             p_pow *= (B_alpha + i) / (B_alpha + B_beta + i)
#             # w_neg *= (B_alpha + i) / (B_alpha + B_beta + i)
#         # pe = B_alpha / (B_alpha + B_beta)
#         # torch.where(pt <= self.clip)
#         # w_neg[torch.where(pe <= self.clip)] = 0
#         # print('pos', w_pos[0])
#         # print('neg', w_neg[0])
#         torch.set_grad_enabled(True)
#         return w_pos * y * Lp + w_neg * (1-y) * Ln



# class EDL_Loss_Test6(nn.Module):
#     def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2, k=2):
#         super(EDL_Loss_Test6, self).__init__()
#         self.k = k
#         self.clip = clip
#         self.gamma_pos = gamma_pos
#         self.gamma_neg = gamma_neg

#     def forward(self, B_alpha, B_beta, y):
        
#         Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
#         Ln = torch.digamma(B_alpha + B_beta + self.gamma_neg) - torch.digamma(B_beta)

#         torch.set_grad_enabled(False)
#         w_pos = torch.ones_like(B_alpha, dtype=float)
#         w_neg = torch.zeros_like(B_beta, dtype=float)
#         p_pow = torch.ones_like(B_alpha, dtype=float)
#         c = torch.tensor(self.clip)
#         # print(c.shape, c.dtype)
#         m = torch.min(c, B_alpha / (B_alpha + B_beta))
#         # w_pos = w_neg = 1
#         for i in range(self.gamma_pos):
#             w_pos *= (B_beta + i) / (B_alpha + B_beta + i)
#         lmd = self.gamma_neg
#         for i in range(lmd+1):
#             w_neg += comb(lmd, i) * torch.pow(-m, lmd-i) * p_pow
#             p_pow *= (B_alpha + i) / (B_alpha + B_beta + i)
#             # w_neg *= (B_alpha + i) / (B_alpha + B_beta + i)
#         # pe = B_alpha / (B_alpha + B_beta)
#         # torch.where(pt <= self.clip)
#         # w_neg[torch.where(pe <= self.clip)] = 0
#         # print('pos', w_pos[0])
#         # print('neg', w_neg[0])
#         torch.set_grad_enabled(True)
#         return w_pos * torch.pow(y, self.k) * Lp + w_neg * torch.pow(1-y, self.k) * Ln


class AsymmetricBetaLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.2, k=2):
        super(AsymmetricBetaLoss, self).__init__()
        self.k = k
        self.clip = clip
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, B_alpha, B_beta, y):
        
        Lp = torch.digamma(B_alpha + B_beta + self.gamma_pos) - torch.digamma(B_alpha)
        m = torch.tensor(self.clip)
        B_alpha_m = torch.max(B_alpha-m/(1-m)*B_beta, torch.zeros_like(B_alpha))
        Ln = torch.digamma(B_alpha_m + B_beta + self.gamma_neg) - torch.digamma(B_beta)

        torch.set_grad_enabled(False)
        w_pos = torch.ones_like(B_alpha, dtype=float)
        w_neg = torch.ones_like(B_beta, dtype=float)
        for i in range(self.gamma_pos):
            w_pos *= (B_beta + i)/ (B_alpha + B_beta + i)
        for i in range(self.gamma_neg):
            w_neg *= (B_alpha_m + i) / (B_alpha_m + B_beta + i)
        torch.set_grad_enabled(True)
        return w_pos * torch.pow(y, self.k) * Lp + w_neg * torch.pow(1-y, self.k) * Ln


# def edl_loss(B_alpha, B_beta, targets):
#     loss = (targets * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_alpha)) 
#             + (1 - targets) * (torch.digamma(B_alpha + B_beta) - torch.digamma(B_beta)))
#     return loss


def uncertainty(B_alpha, B_beta, num_class):
    return 2 * num_class / np.sum(B_alpha, axis=1)


# def uncertainty_neg(B_alpha, B_beta, num_class, tau=1):
#     return num_class / np.sum(B_beta, axis=1)


# def uncertainty_posneg(B_alpha, B_beta, num_class):
#     return 2 * num_class / np.sum(B_alpha + B_beta, axis=1)
