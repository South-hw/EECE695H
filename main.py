import os
import argparse
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.dataset import CUB as Dataset
from src.sampler import Sampler
from src.train_sampler import Train_Sampler
from src.utils import count_acc, Averager
from model import FewShotModel, CosLinear
from losses import MixupLoss
from trainer import Trainer

def train(args):
    # 1. load dataset
    train_dataset = Dataset(args.dpath, state='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch,
            shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = Dataset(args.dpath, state='val')
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch,
            shuffle=False, num_workers=4, pin_memory=True)


    """ TODO 1.a """
    " Make your own model for Few-shot Classification in 'model.py' file."

    # model setting
    if args.cont:
        model = FewShotModel()
        model.load_state_dict(torch.load(args.ckpt, map_location='cuda:0'))
        model = model.cuda().train()
    else:
        model = FewShotModel().cuda().train()
    classifier_rot = nn.Sequential(nn.Linear(640, 4)).cuda().train()

    """ TODO 1.a END """

    """ TODO 1.b (optional) """
    " Set an optimizer or scheduler for Few-shot classification (optional) "

    # Default optimizer setting
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': classifier_rot.parameters()}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
            step_size=5, gamma=0.5)

    """ TODO 1.b (optional) END """
    loss_fn_mixup = MixupLoss()
    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_rot = nn.CrossEntropyLoss()

    writer = SummaryWriter('../outputs/tensorboard/train2')

    trainer = Trainer(net=model, classifier_rot=classifier_rot, 
            train_loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, loss_fn_mixup=loss_fn_mixup, 
            loss_fn_cls=loss_fn_cls, loss_fn_rot=loss_fn_rot, 
            writer=writer, save_root=args.save_root)

    # training start
    for e in range(args.epoch):
        trainer.fit()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str,
            default='../outputs/weights/base')
    parser.add_argument('--dpath', '--d', default='./dataset/CUB_200_2011/CUB_200_2011', type=str,
                        help='the path where dataset is located')
    parser.add_argument('--gpus', type=str, default='1')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--cont', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train(args)

