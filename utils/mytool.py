import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional

def select_confident_region(out1,out2,noise_label,thed=0.5):

    out1_soft = torch.softmax(out1,1)
    out2_soft = torch.softmax(out2,1)
    
    label1 = torch.argmax(out1_soft,1)
    label2 = torch.argmax(out2_soft,1)
    
    # assessment uncertainty for cnn1
    logit1, _ = torch.topk(out1_soft, k=2, dim=1)
    logit_max1 = logit1[:, 0]
    logit_sec1 = logit1[:, 1]
    un1 = -(logit_max1 * torch.log(logit_max1) + logit_sec1 * torch.log(logit_sec1))

    # assessment uncertainty for cnn2
    logit2, _ = torch.topk(out2_soft, k=2, dim=1)
    logit_max2 = logit2[:, 0]
    logit_sec2 = logit2[:, 1]
    un2 = -(logit_max2 * torch.log(logit_max2) + logit_sec2 * torch.log(logit_sec2))

    cross_mask1 = (un1<un2)#&(logit_max1>0.90)
    cross_mask2 = (un2<un1)#&(logit_max2>0.90)
    
    mix_label = torch.zeros_like(noise_label).long().cuda()
    
    mix_label[cross_mask1==True] = label1[cross_mask1==True]
    mix_label[cross_mask2==True] = label2[cross_mask2==True]    
    
    low_region = (logit_max1<thed) & (logit_max2<thed)
    
    mix_label[low_region==True] = 255
    
    return mix_label

def merge_cnn_without_confidence(out1,out2):
    out1_soft = torch.softmax(out1,1)
    out2_soft = torch.softmax(out2,1)
    
    merge_label = torch.argmax((out1_soft+out2_soft)/2,1)
    
    return merge_label


def eval_image(predict,label,num_classes):
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)
                   

def structure_loss(pred, mask,num_classes):
    ce_loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=255)
    wbce = ce_loss(pred,mask.squeeze(1).long())
    dice_loss = DiceLoss(num_classes)
    dice = dice_loss(pred,mask)

    return wbce+dice

def entropy_loss(p1,p2, p3,C=2):
    p1 = torch.softmax(p1,0)
    p2 =  torch.softmax(p2,0)
    p3 = torch.softmax(p3,0)
    p = (p1+p2+p3)/3
    
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
   
    ent = torch.mean(y1)

    return ent

def joint_optimization(outputs_main, outputs_aux1, outputs_aux2,kd_T):
    kd_loss = KDLoss(T=kd_T)
    avg_aux = (outputs_aux1 + outputs_aux2) / 2

    L_kd = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                   avg_aux.permute(0, 2, 3, 1).reshape(-1, 4))
    return L_kd

def label_accuracy_score(label_trues, label_preds, n_class=19):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i) & (input_tensor != 255)  # 忽略值为 255 的区域
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)

        if z_sum + y_sum == 0:  # 避免除以零
            return torch.tensor(1.0, device=score.device)  # 返回最大损失

        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        target = target.squeeze(2)
        
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        return loss / self.n_classes


def dice_loss(pred, target, smooth=1e-6):
    # 将标签值为 255 的区域设置为 -1
    target = target.clone()
    target[target == 255] = -1  # 选择一个负值来表示忽略区域

    total_loss = 0.0
    for i in range(pred.shape[1]):  # 遍历每个类别
        pred_i = pred[:, i]  # 选择当前类别的预测
        target_i = target.clone()
        target_i[target_i != i] = 0  # 只保留当前类别

        # 只保留有效的预测和目标
        valid_mask = (target_i != -1)
        pred_i = pred_i[valid_mask]
        target_i = target_i[valid_mask]

        # 计算 Dice 系数
        intersection = (pred_i * target_i).sum()
        pred_sum = pred_i.sum()
        target_sum = target_i.sum()

        if pred_sum > 0 and target_sum > 0:  # 确保有有效元素
            dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
            total_loss += (1 - dice)
        else:
            total_loss += 1  # 如果没有有效元素，可以给一个默认的损失

    return total_loss / pred.shape[1]  # 返回平均 Dice 损失



def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                     F.softmax(out_t / self.T, dim=1), reduction="mean") # , reduction="batchmean"
            * self.T
            * self.T
        )
        return loss


class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.05, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )