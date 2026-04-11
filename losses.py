import numpy as np

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Criterion(nn.Module):
    """
    Compute loss and evaluation metrics
    """
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.args = args
        self.weights = args.loss_weight
        self.l1loss = nn.L1Loss()

    def forward(self, preds, true):
        loss_l1 = 0.
        loss_l1 += 0.5*self.l1loss(preds[0], true)
        loss_l1 += 1.0*self.l1loss(preds[1], true)

        """ -------------------- aggregate losses -------------------- """
        l_sum = self.weights[0]*loss_l1
        loss_list = [self.weights[0]*loss_l1]

        return l_sum, loss_list

class Criterion_twostage(nn.Module):
    """
    Compute loss and evaluation metrics
    """
    def __init__(self, args):
        super(Criterion_twostage, self).__init__()
        self.args = args
        self.weights = args.loss_weight
        self.l1loss = nn.L1Loss()
    
    def calc_psnr_loss(self, preds, true):
        scale = 10 / np.log(10)
        loss = 0.
        for pred in preds:
            loss += scale * torch.log(((pred - true) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        return loss

    def forward(self, preds, true):
        loss_l1 = 0.
        loss_l1 += 1.0*self.l1loss(preds[0], true)

        """ -------------------- aggregate losses -------------------- """
        l_sum = self.weights[0]*loss_l1
        loss_list = [self.weights[0]*loss_l1]

        return l_sum, loss_list