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
        # TODO convert the embeddings to onehot + linear
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bpr_loss = BPRLoss()
        self.dcg_loss = SmoothDCGLoss(device=self.device, topk=50, temp=self.temp)
        # parameters initialization
        self.apply(xavier_normal_initialization)

        # TODO how is the device set in recbole?
        self.device = config["device"]
        # TODO add temperature to the config
        self.temp = config["temperature"]
        self.dataset_name = dataset.dataset_name

    def set_max_pos(self, train_data, val_data):
        # TODO check if this definition of max_length
        #  (adaption of haolun to recbole)
        # works

        # begin
        # train_val_user_list[i] contains the list of items
        # consumed by user i, either in the train or in the
        # val set
        counts_per_user = train_data[self.USER_ID].value_counts() + val_data[self.USER_ID].value_counts()
        max_length = counts_per_user.max()
        print("max_train_val_length:", max_length)
        if self.dataset_name == 'ml-1m' or 'ml-100k':
            max_pos = max_length if max_length < 200 else 200
        elif self.dataset_name == 'lastfm':
            max_pos = max_length if max_length < 100 else 100
        print("max_pos:", max_pos)
        self.max_pos = max_pos


    def get_max_pos(self, ):
        # TODO
        # Sample max_pos items for each user.
        # Those are the items to be used in the NDCG approx loss (fixed number).
        # All other items are ignored for NDCG loss.
        # If less than max_pos items are available
        # The remaining items are taken from the negative ones
        self.neg_ids_list = []
        self.pos_ids_list = []

        user_size = self.n_users
        item_size = self.n_items
        # TODO train_user_list[i] contains the list of
        # items consumed by user i according to the
        # train set
        for i in range(user_size):
            if (len(train_user_list[i]) > max_pos):
                sampled_pos_ids = np.random.choice(len(train_user_list[i]), size=max_pos, replace=False)
                tmp = [train_user_list[i][j] for j in sampled_pos_ids]
                self.pos_ids_list.append(tmp)
            else:
                self.pos_ids_list.append(train_user_list[i])
            self.neg_ids_list.append(negsamp_vectorized_bsearch_preverif(np.array(train_user_list[i]), item_size,
                                                                    n_samp=max_pos - len(pos_ids_list[i])))

        self.sampled_ids = np.ones((user_size, max_pos)) * item_size
        self.labels = np.zeros((user_size, max_pos))

        for i in range(user_size):
            self.sampled_ids[i][:len(self.pos_ids_list[i])] = np.array(self.pos_ids_list[i])
            self.sampled_ids[i][len(self.pos_ids_list[i]):] = self.neg_ids_list[i]
            self.labels[i][:len(self.pos_ids_list[i])] = 1

        self.sampled_ids = torch.LongTensor(self.sampled_ids).to(args.device)
        self.labels = torch.LongTensor(self.labels).to(args.device)

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
        # TODO adapt in the case of a linear layer? (i.e., are the embeddings still simply the weights?)
        all_item_e = self.item_embedding.weight
        scores_all = torch.mul(user_e, pos_e).sum(dim=1)
        scores = torch.gather(scores_all, 1, sampled_ids).to(args.device)

        unique_u = torch.LongTensor(list(set(user.tolist())))

        # TODO define the labels
        # TODO check if the right users are accessed by scores(_all)[unique_u]
        # dcg_loss = self.dcg_loss(scores_top=scores[unique_u], scores=scores_all[unique_u], labels=labels[unique_u])
        # TODO separate the two classes(e.g., male and female)
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
