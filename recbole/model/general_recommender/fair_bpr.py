# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.model.loss import SmoothDCGLoss
from recbole.utils import InputType


class BPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bpr_loss = BPRLoss()
        self.dcg_loss = SmoothDCGLoss(device=self.device, topk=50, temp=self.temp)
        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.device = config["device"]
        self.temp = config["temperature"]

    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        # User is a batch of (possibly repeating)
        # user ids
        # Item is a batch of (possibly repeating)
        # item ids coresponding to the interactions of the above users
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        # This is a batch of possibly repeating user ids
        user = interaction[self.USER_ID]
        # This is the batch of items with which the user interacted
        # (positive or negative)
        pos_item = interaction[self.ITEM_ID]
        # One negative interaction for each positive interaction
        neg_item = interaction[self.NEG_ITEM_ID]

        # Embed the users and positive items
        user_e, pos_e = self.forward(user, pos_item)
        # Embed the negative items
        neg_e = self.get_item_embedding(neg_item)
        # pos_item_score and neg_item_score will be vectors of the same length
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        bpr_loss = self.bpr_loss(pos_item_score, neg_item_score)
        # TODO
        # See localhost:8888/lab/tree/Multi-Fair-Rec/main_bpr.py
        all_item_e = self.item_embedding.weight
        scores_all = torch.matmul(user_e, all_item_e.transpose(0, 1))
        unique_u = torch.LongTensor(list(set(user.tolist())))

        #  def forward(self, scores_top, scores, labels):
        # dcg_loss = self.dcg_loss()
        dcg_loss = 0.

        return bpr_loss + dcg_loss

    def predict(self, interaction):
        # TODO check with alessandro!!
        # Does it only predict for the items that are in the batch??
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight

        # Reduce memory usage
        # user_e, all_item_e = user_e.type(torch.bfloat16), all_item_e.type(torch.bfloat16)
        # print(user_e.type(), all_item_e.type())
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
