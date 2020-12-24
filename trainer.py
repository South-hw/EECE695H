import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, **kwarge):
        self.net = kwarge['net']
        self.classifier_rot = kwarge['classifier_rot']

        self.train_loader = kwarge['train_loader']
        self.optimizer = kwarge['optimizer']

        self.loss_fn_mixup = kwarge['loss_fn_mixup']
        self.loss_fn_cls = kwarge['loss_fn_cls']
        self.loss_fn_rot = kwarge['loss_fn_rot']

        self.writer = kwarge['writer']
        self.save_root = kwarge['save_root']

        self.cur_epoch = 0
        self.cur_iter = 0

    def fit(self):
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                desc=f'train_{self.cur_epoch}')

        for i, (imgs, labels) in pbar:
            imgs = imgs.cuda()
            labels = labels.cuda()

            # 1. manifold mixup
            f, pred_cls, target_a, target_b, lam = self.net(imgs, target=labels, mixup=True)
            loss_mm = self.loss_fn_mixup(pred_cls, target_a, target_b, lam)

            # 2. rotation & cls
            bs = imgs.shape[0]
            inputs_ = []
            targets_ = []
            a_ = []
            idx = torch.randperm(bs)

            split_size = int(bs/4)
            for j in idx[:split_size]:
                x90 = imgs[j].transpose(2, 1).flip(1)
                x180 = x90.transpose(2, 1).flip(1)
                x270 = x180.transpose(2, 1).flip(1)
                inputs_ += [imgs[j], x90, x180, x270]
                targets_ += [labels[j] for _ in range(4)]
                a_ += [torch.tensor(0), torch.tensor(1), torch.tensor(2),
                        torch.tensor(3)]
            inputs = torch.stack(inputs_, 0).detach().clone().cuda()
            targets = torch.stack(targets_, 0).detach().clone().cuda()
            a = torch.stack(a_, 0).detach().clone().cuda()

            f, pred_cls = self.net(inputs)
            pred_rot = self.classifier_rot(f)

            loss_cls = self.loss_fn_cls(pred_cls, targets)
            loss_rot = self.loss_fn_rot(pred_rot, a)

            loss = loss_mm + (loss_cls + loss_rot) * 0.5

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.writer.add_scalar('loss/loss', loss.item(), self.cur_iter)
            self.writer.add_scalar('loss/loss_mm', loss_mm.item(), self.cur_iter)
            self.writer.add_scalar('loss/loss_rot', loss_rot.item(), self.cur_iter)
            self.writer.add_scalar('loss/loss_cls', loss_cls.item(), self.cur_iter)
            self.cur_iter += 1
        
        torch.save(self.net.state_dict(), os.path.join(self.save_root,
            f'net_{self.cur_epoch}.pth'))
        self.cur_epoch += 1






