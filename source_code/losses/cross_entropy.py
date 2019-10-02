import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from lib.config import cfg


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        loss = F.nll_loss(F.log_softmax(x, dim=1), y)
        return loss, ('01. cross_entropy loss: ', loss.data.cpu().numpy())


class SmoothCrossEntropy(nn.Module):
    def __init__(self):
        super(SmoothCrossEntropy, self).__init__()
        self.eta = cfg.LOSSES.LABEL_SMOOTH

    def forward(self, x, y):
        class_num = x.shape[1]
        pos = 1 - self.eta + self.eta / class_num
        neg = self.eta / class_num
        one_hot = (neg * torch.ones(x.shape).type(x.type())).scatter_(1, torch.unsqueeze(y, dim=1), pos)
        per_sample_loss = (-1. * F.log_softmax(x, dim=1) * one_hot).sum(dim=1)
        loss = per_sample_loss.mean()
        return loss, ('01. smooth cross_entropy loss: ', loss.data.cpu().numpy())
