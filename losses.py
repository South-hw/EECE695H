import torch
import torch.nn as nn
import torch.nn.functional as F


class MixupLoss(nn.Module):
    def __init__(self):
        super(MixupLoss, self).__init__()
        self.loss_fn1 = nn.CrossEntropyLoss()
        self.loss_fn2 = nn.CrossEntropyLoss()

    def forward(self, x, y_a, y_b, lam):
        loss1 = self.loss_fn1(x, y_a)
        loss2 = self.loss_fn2(x, y_b)
        loss = lam * loss1 + (1 - lam) * loss2
        return loss
