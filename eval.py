import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from model import FewShotModel
from processing import Processing
import pdb


def eval(epoch):
    dpath = '/data/CUB_200_2011/'

    val_dataset = Dataset(dpath, state='val')
    val_sampler = Sampler(val_dataset._labels, n_way=5, k_shot=5, query=20)
    val_data_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
            num_workers=4, pin_memory=True)

    pth_path = f'../outputs/weights/base/net_{epoch}.pth'
    net = FewShotModel()
    net.load_state_dict(torch.load(pth_path, map_location='cuda:0'))
    net = net.cuda()

    acc = 0.0
    for _ in range(200):
        for imgs, labels in val_data_loader:
            imgs = imgs.cuda()

            with torch.no_grad():
                features, _ = net(imgs)
            _, labels = torch.unique(labels, sorted=True, return_inverse=True)
            P = Processing(features=features.cpu(), labels=labels)
            acc += P.map()

            del P
    print(f'epoch: {epoch}, acc: {acc/200}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()

    start = args.start
    end = args.end
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    for i in range(start, end+1):
        eval(i)

