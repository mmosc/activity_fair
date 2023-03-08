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
import pandas as pd
import torch.nn.functional as F

import numpy as np

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.model.loss import SmoothDCGLoss
from recbole.utils import InputType
from recbole.model.model_utils import negsamp_vectorized_bsearch_preverif


class ToOneHot(nn.Module):
    def __init__(self, num_classes):
        super(ToOneHot, self).__init__()
        self.num_classes = num_classes
    def forward(self, x):
        return F.one_hot(torch.LongTensor([x], num_classes=self.num_classes))



class FairBPR(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(FairBPR, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        # TODO reproducibility seed for max_pos
        # TODO convert the embeddings to onehot + linear
        """
        self.user_embedding = nn.Sequential(
            ToOneHot(self.n_users),
            nn.Linear(self.n_users, self.embedding_size),
        )
        nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Sequential(
            ToOneHot(self.n_items),
            nn.Linear(self.n_items, self.embedding_size),
        )"""
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.bpr_loss = BPRLoss()
        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.temp = config["temperature"]
        self.dataset_name = dataset.dataset_name
        self.dcg_loss = SmoothDCGLoss(device=self.device, topk=50, temp=self.temp)

        self.item_set = torch.range(start=1, end=self.n_items - 1, step=1, dtype=int)

        self.get_token2gender('/home/marta/jku/fairinterplay/dataset/ml-100k/ml-100k.user')
        self.dataset = dataset

    def get_token2gender(self, user_info_path):
        """
        sets the Dict of the token id (i.e., original dataset id)
        to gender.
        REMARK!! This is NOT the recbole ID
        Args:
            user_info_path:

        Returns:

        """
        # https://github.com/haolun-wu/Multi-Fair-Rec/blob/f4463f017f1691841743122fe4ffa379aa2186e6/preprocess.py#L29
        user_info_df = pd.read_csv(user_info_path, sep='\t')
        user_info_df = user_info_df[['user_id:token', 'gender:token']]
        self.token2gender = {str(row[1][0]): row[1][1] for row in user_info_df.iterrows()}

    def set_max_pos(self, train_data, val_data):
        """
        Sets the self.max_pos variable either to the maximum
        number of interactions of a single user, considering
        train + val, or with the same threshold used by
        haolun for ml and lastfm.

        Args:
            train_data: train data in RecBole format
            val_data: val data in RecBole format

        Returns:

        """
        self.train_count_dict = {int(user): int(counts) for user, counts in
                                 torch.stack(train_data.dataset[self.USER_ID].unique(return_counts=True)).T}

        self.val_count_dict = {int(user): int(counts) for user, counts in
                               torch.stack(val_data.dataset[self.USER_ID].unique(return_counts=True)).T}
        train_val_counts = {key: self.train_count_dict.get(key, 0) + self.val_count_dict.get(key, 0)
                            for key in set(self.train_count_dict) | set(self.val_count_dict)}

        max_length = max(train_val_counts, key=train_val_counts.get)
        if self.dataset_name == 'ml-1m' or 'ml-100k':
            max_pos = max_length if max_length < 200 else 200
        elif self.dataset_name == 'lastfm':
            max_pos = max_length if max_length < 100 else 100
        self.max_pos = max_pos

    def get_max_pos(self, train_data, val_data):
        # TODO check the adaption of haolun

        # sampled_ids seems to be defined outside the batch iterations.
        # code for neg_ids_list and pos_ids_list to recbole
        #
        # What it should do:
        # Sample self.max_pos items for each user.
        # Those are the items to be used in the NDCG approx loss (fixed number).
        # All other items are ignored for NDCG loss.
        # If less than max_pos items are available
        # The remaining items are taken from the negative ones
        self.neg_ids_dict = {}
        self.pos_ids_dict = {}

        self.set_max_pos(train_data, val_data)
        for user_index, train_counts in self.train_count_dict.items():
            user_pos = train_data.dataset[train_data.dataset[self.USER_ID] == user_index][self.ITEM_ID]
            if train_counts > self.max_pos:
                sampled_pos_ids = np.random.choice(train_counts, size=self.max_pos, replace=False)
                tmp = [user_pos[j] for j in sampled_pos_ids]
                self.pos_ids_dict[user_index] = tmp
            else:
                self.pos_ids_dict[user_index] = list(user_pos)

            self.neg_ids_dict[user_index] = negsamp_vectorized_bsearch_preverif(np.array(user_pos), self.n_items,
                                                                                n_samp=self.max_pos - len(
                                                                                    self.pos_ids_dict[user_index]))

        train_user_ids = self.train_count_dict.keys()
        self.sampled_ids = {user_id: np.ones(self.max_pos) * self.n_items for user_id in train_user_ids}
        self.labels = {user_id: np.zeros(self.max_pos) for user_id in train_user_ids}

        # REMARK in my version, self.sample_ids are dictionaries,
        # but they should probably be converted to tensors
        # This requires a proper identification of the
        # user ids / recbole ids / ....
        for user_id in train_user_ids:
            self.sampled_ids[user_id][:len(self.pos_ids_dict[user_id])] = self.pos_ids_dict[user_id]
            self.sampled_ids[user_id][len(self.pos_ids_dict[user_id]):] = self.neg_ids_dict[user_id]
            self.sampled_ids[user_id] = self.sampled_ids[user_id].astype(int)
            self.labels[user_id][:len(self.pos_ids_dict[user_id])] = 1

        self.sampled_ids_tensor_noPAD = torch.tensor(
            np.array([sampled_ids_user - 1 for sampled_ids_user in self.sampled_ids.values()]))
        self.sampled_labels_tensor = torch.tensor(
            np.array([sampled_labels_user for sampled_labels_user in self.labels.values()]))

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

        # REMARK: This way we are only scoring the items in the batch.
        # See localhost:8888/lab/tree/Multi-Fair-Rec/main_bpr.py
        # The set of users in this batch
        batch_user_set_ids = torch.unique(user)
        # batch_user_genders = [self.userid_to_gender[user_id]
        batch_user_e, all_item_e = self.forward(batch_user_set_ids, self.item_set)
        #print(batch_user_e.shape, all_item_e.shape)
        scores_all = torch.matmul(batch_user_e, all_item_e.T)

        # The embedding of the max_pos sampled ids of each user (in the batch)
        # This should be restricted to the users in the batch

        batch_sampled_ids_tensor_noPAD = self.sampled_ids_tensor_noPAD[batch_user_set_ids - 1]
        scores = torch.gather(scores_all, 1, batch_sampled_ids_tensor_noPAD).to(self.device)
        # unique_u = torch.LongTensor(list(set(user.tolist())))
        batch_sampled_labels_tensor = self.sampled_labels_tensor[batch_user_set_ids - 1]

        ##########
        # Remark this is restricted to the batch, i.e., be careful with the indices when selecting male and female
        ##########
        # batch_userid_to_gender = {user_id: self.token2gender[interaction.id2token(interaction.uid_field, user_id)] for user_id in batch_sampled_ids_tensor_noPAD}
        # List of genders ordered as in the current batch
        batch_userid_gender = [self.token2gender[self.dataset.id2token(self.dataset.uid_field, user_id)] for user_id in batch_user_set_ids]
        mask_F = [gender == 'F' for gender in batch_userid_gender]
        mask_M = [gender == 'M' for gender in batch_userid_gender]

        ndcg = self.dcg_loss(scores_top=scores, scores=scores_all, labels=batch_sampled_labels_tensor)

        ndcg_F = ndcg[mask_F]
        ndcg_M = ndcg[mask_M]

        # sum the matrix in column
        ndcg_F = ndcg_F.sum(dim=0) / sum(mask_F)
        ndcg_M = ndcg_M.sum(dim=0) / sum(mask_M)

        fairness_loss = torch.abs(torch.log(1 + torch.abs(ndcg_M[-1] - ndcg_F[-1]))).sum()

        return bpr_loss + fairness_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
