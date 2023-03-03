# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import math


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class SmoothRank(torch.nn.Module):
    """
    See
    https://github.com/haolun-wu/Multi-Fair-Rec/blob/main/SoftRank.py
    """

    def __init__(self, temp=1):
        """

        Args:
            temp: temperature. how soft the ranks to be
        """
        super(SmoothRank, self).__init__()
        self.temp = temp
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, scores_max_relevant, scores):
        x_0 = scores_max_relevant.unsqueeze(dim=-1)
        x_1 = scores.unsqueeze(dim=-2)
        diff = x_1 - x_0
        is_lower = diff / self.temp
        is_lower = self.sigmoid(is_lower)
        # del diff

        ranks = torch.sum(is_lower, dim=-1) + 0.5
        # del is_lower
        # torch.cuda.empty_cache()
        return ranks


class SmoothDCGLoss(nn.Module):
    """
    See
    https://github.com/haolun-wu/Multi-Fair-Rec/blob/main/SoftRank.py
    """

    def __init__(self, device, topk, temp=1):
        """

        Args:
            args:
            topk:
            temp: temperature. how soft the ranks to be
        """
        super(SmoothDCGLoss, self).__init__()
        self.smooth_ranker = SmoothRank(temp)
        self.zero = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.one = nn.Parameter(torch.tensor([1], dtype=torch.float32), requires_grad=False)
        self.topk = topk
        # TODO
        self.device = device
        self.idcg_vector = self.idcg_k()

    def idcg_k(self):
        res = torch.zeros(self.topk).to(self.device)

        for k in range(1, self.topk + 1):
            res[k - 1] = sum([1.0 / math.log(i + 2, 2) for i in range(k)])

        return res

    def forward(self, scores_top, scores, labels):
        ranks = self.smooth_ranker(scores_top, scores)
        # print("ranks:", ranks)
        d = torch.log2(ranks + 1)
        dg = labels / d

        ndcg = None

        for p in range(1, self.topk + 1):
            dg_k = dg[:, :p]
            dcg_k = dg_k.sum(dim=-1)
            k = torch.sum(labels, dim=-1).long()
            k = torch.clamp(k, max=p, out=None)
            ndcg_k = (dcg_k / self.idcg_vector[k - 1]).reshape(-1, 1)

            ndcg = ndcg_k if ndcg is None else torch.cat((ndcg, ndcg_k), dim=1)

        # print("ndcg:", ndcg.shape)

        # dcg = dg.sum(dim=-1)
        # k = torch.sum(labels, dim=-1).long()
        # k = torch.clamp(k, max = self.topk, out=None)
        # dcg = dcg / self.idcg_vector[k-1]
        # dcg = dcg

        return ndcg


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding ** self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
