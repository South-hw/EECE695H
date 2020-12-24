import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
""" Optional conv block """
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        bias=True):
    block = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias)
    )
    return block



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, stride=1):
        super(BasicBlock, self).__init__()
        self.block1 = conv_block(in_channels=in_channels,
                out_channels=out_channels, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.block2 = conv_block(in_channels=out_channels,
                out_channels=out_channels, stride=stride, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=1, stride=stride, bias=False)
            )
 
    def forward(self, x):
        out = self.block1(x)
        out = self.dropout(out)
        out = self.block2(out)

        out += self.shortcut(x)
        return out


""" Define your own model """
class FewShotModel(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3,
            num_classes=100):
        super(FewShotModel, self).__init__()
        self.in_planes = 16
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1,
                padding=1, bias=False)
        self.layer1 = self._wide_layer(BasicBlock, nStages[1], n, dropout_rate, 1)
        self.layer2 = self._wide_layer(BasicBlock, nStages[2], n, dropout_rate, 2)
        self.layer3 = self._wide_layer(BasicBlock, nStages[3], n, dropout_rate, 2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = CosLinear(nStages[3], 200)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _mixup(self, x, y):
        b = x.shape[0]
        idx = torch.randperm(b).cuda()

        lam = np.random.beta(2.0, 2.0)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        y_a, y_b = y, y[idx]
        return mixed_x, y_a, y_b, lam

    def forward(self, x, target=None, mixup=False):
        if mixup:
            return self.forward_mixup(x=x, target=target)
        else:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            out1 = self.linear(out)

            return out, out1

    def forward_mixup(self, x, target):
        layer_mix = torch.randint(low=0, high=3, size=(1, ))

        out = x
        target_a = target_b = target

        if layer_mix == 0:
            out, target_a, target_b, lam = self._mixup(x=out, y=target)
        out = self.conv1(out)

        out = self.layer1(out)
        if layer_mix == 1:
            out, target_a, target_b, lam = self._mixup(x=out, y=target)

        out = self.layer2(out)
        if layer_mix == 2:
            out, target_a, target_b, lam = self._mixup(x=out, y=target)

        out = self.layer3(out)
        if layer_mix == 0:
            out, target_a, target_b, lam = self._mixup(x=out, y=target)

        out = F.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out1 = self.linear(out)
        return out, out1, target_a, target_b, lam


class CosLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(CosLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) 
        scores = self.scale_factor* (cos_dist) 

        return scores   
 



