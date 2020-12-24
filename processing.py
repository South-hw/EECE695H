import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Processing:
    def __init__(self, features, labels, beta=0.5, lam=10, alpha=0.2, n_way=5, k_shot=5, query=20):
        # (n_way * k_shot, 640)
        self.beta = beta
        self.lam = lam
        self.alpha = alpha

        self.n_way = n_way
        self.k_shot = k_shot

        self.n_ss = n_way * k_shot
        self.n_q = query

        self.features = features
        self.labels = labels

        self._pt()
        self._normalization()
        self._initialization()

    def _pt(self):
        self.features = torch.pow(self.features + 1e-6, self.beta)
        self.features = torch.qr(self.features.permute(1, 0)).R
        self.features = self.features.permute(1, 0)

        norm = torch.norm(self.features, dim=1, keepdim=True)

        self.features /= norm

    def _qr_reudction(self):
        self.features = torch.qr(self.features.permute(1, 0)).R
        self.features = self.features.permute(1, 0)

 
    def _normalization(self):
        self.features[:self.n_ss, :] -= torch.mean(self.features[:self.n_ss,:], 
                dim=0, keepdim=True)
        self.features[:self.n_ss, :] /= torch.norm(self.features[:self.n_ss,:], 
                p=2, dim=1, keepdim=True)
        self.features[self.n_ss:, :] -= torch.mean(self.features[self.n_ss:,:], 
                dim=0, keepdim=True)
        self.features[self.n_ss:, :] /= torch.norm(self.features[self.n_ss:,:], 
                p=2, dim=1, keepdim=True)

    def _initialization(self):
        #self.mus = self.features[:self.n_ss].reshape(self.k_shot, self.n_way, -1).mean(dim=0)
        self.mus = self.features[:self.n_ss].reshape(self.n_way, self.k_shot, -1).mean(dim=1)

    def _get_prob(self):
        dist = (self.features.unsqueeze(1) - \
                self.mus.unsqueeze(0)).norm(dim=2).pow(2)

        p_xj = torch.zeros_like(dist)
        r = torch.ones((self.n_q, ))
        c = torch.ones((self.n_way)) * self.n_q

        p_xj_test, _ = self.compute_optimal_transport(dist[self.n_ss:], r, c)
        p_xj[self.n_ss:] = p_xj_test

        p_xj[:self.n_ss].fill_(0)
        p_xj[:self.n_ss].scatter_(1, self.labels[:self.n_ss].unsqueeze(1), 1)

        return p_xj

    def compute_optimal_transport(self, M, r, c, eps=1e-6):
        n, m = M.shape

        P = torch.exp(-self.lam * M)
        P /= P.sum()

        u = torch.zeros((n, ))
        max_iter = 1000
        iters = 1
        while torch.max(torch.abs(u - P.sum(1))) > eps:
            u = P.sum(1)
            P *= (r / u).view(-1, 1)
            P *= (c / P.sum(0)).view(1, -1)
            
            if iters == max_iter:
                break
            iters += 1
        return P, torch.sum(P * M)

    def estimateFromMask(self, mask):
        emus = mask.permute(1,
                0).matmul(self.features).div(mask.sum(dim=0).unsqueeze(1))
        return emus
    
    def updateFromEstimate(self, estimate):
        Dmus = estimate - self.mus
        self.mus = self.mus + self.alpha * (Dmus)

    def getAcc(self, probas):
        pred = probas.argmax(dim=1)
        matches = (pred == self.labels).float()

        acc = matches[self.n_ss:].mean()

        return acc

    def map(self, epoch=20):
        for i in range(epoch):
            p_xj = self._get_prob()
            self.probas = p_xj

            m_estimate = self.estimateFromMask(self.probas)
            self.updateFromEstimate(m_estimate)


        p_xj = self._get_prob()
        self.probas = p_xj

        acc = self.getAcc(probas=self.probas)
        return acc
