# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/model/base/reco_abc.py
#

import numpy as np
import random
import logging
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_

import recommenders.models.unirec.model.modules as modules
from recommenders.models.unirec.constants.loss_funcs import LossFuncType
from recommenders.models.unirec.constants.global_variables import EPS, VALID_TRIGGER_P
from recommenders.models.unirec.constants.protocols import DataFileFormat
from recommenders.models.unirec.utils import file_io


def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.0)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def normal_initialization(mean, std):
    def normal_init(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return normal_init


def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.0)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class AbstractRecommender(nn.Module):
    def __init__(self, config):
        super(AbstractRecommender, self).__init__()
        self.logger = logging.getLogger(config["exp_name"])
        self.__optimized_by_SGD__ = True
        self.config = config
        self._init_attributes()
        self._init_modules()

        self.annotations = []
        self.add_annotation()

        self._parameter_validity_check()

    ## -------------------------------
    ## Basic functions you need to pay attention to.
    ## In most cases, you will need to override them.
    ## -------------------------------
    def _parameter_validity_check(self):
        # if self.loss_type in [LossFuncType.BPR.value, LossFuncType.CCL.value]:
        #     if self.config['train_file_format'] not in [DataFileFormat.T1.value, DataFileFormat.T2.value, DataFileFormat.T5.value, DataFileFormat.T6.value]:
        #         raise ValueError(r'''
        #                 for efficiency concern, we put the limitation in implementation:
        #                 if you want to use BPR or CCL as the loss function, please make sure only the first item in one group is positive item.
        #                 and the data format is T1 or T4
        #         ''')

        if self.loss_type == LossFuncType.SOFTMAX.value:
            if (
                self.config["train_file_format"]
                in [DataFileFormat.T2.value, DataFileFormat.T2_1.value]
                and self.group_size <= 0
            ):
                raise ValueError(
                    r""" 
                        if you want to use SOFTMAX as the loss function in user-item-label data format,
                        please make sure you have correct the group_size hyper-parameter. 
                        Each positive line should follow by the same number of negative lines.
                """
                )

    def _define_model_layers(self):
        raise NotImplementedError

    def forward(self, user_id):
        raise NotImplementedError

    def forward_user_emb(self, interaction):
        raise NotImplementedError

    def forward_item_emb(self, interaction):
        raise NotImplementedError

    ## -------------------------------
    ## More functions you may need to override.
    ## -------------------------------
    def _predict_layer(self, user_emb, items_emb, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def add_annotation(self):
        self.annotations.append("AbstractRecommender")

    ## -------------------------------
    ## Belowing functions can fit most scenarios so you don't need to override.
    ## -------------------------------

    def _init_attributes(self):
        config = self.config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.device = config["device"]
        self.loss_type = config.get("loss_type", "bce")
        self.embedding_size = config.get("embedding_size", 0)
        self.hidden_size = self.embedding_size
        self.dropout_prob = config.get("dropout_prob", 0.0)
        self.use_pre_item_emb = config.get("use_pre_item_emb", 0)
        self.use_text_emb = config.get("use_text_emb", 0)
        self.text_emb_size = config.get("text_emb_size", 768)
        self.init_method = config.get("init_method", "normal")
        self.use_features = config.get("use_features", 0)
        if self.use_features:
            self.feature_emb_size = (
                self.embedding_size
            )  # config.get('feature_emb_size', 40)
            self.features_shape = eval(config.get("features_shape", "[]"))
            self.item2features = file_io.load_features(
                self.config["features_filepath"], self.n_items, len(self.features_shape)
            )
        if "group_size" in config:
            self.group_size = config["group_size"]
        else:
            self.group_size = -1
        ## clip the score to avoid loss being nan
        ## usually this is not necessary, so you don't need to set up score_clip_value in config
        self.SCORE_CLIP = -1
        if "score_clip_value" in self.config:
            self.SCORE_CLIP = self.config["score_clip_value"]
        self.has_user_bias = False
        self.has_item_bias = False
        if "has_user_bias" in config:
            self.has_user_bias = config["has_user_bias"]
        if "has_item_bias" in config:
            self.has_item_bias = config["has_item_bias"]
        self.tau = config.get("tau", 1.0)

    def _init_modules(self):
        # define layers and loss
        # TODO: remove user_embedding when user_id is not needed to save memory. Like in VAE.
        if self.has_user_bias:
            self.user_bias = nn.Parameter(torch.normal(0, 0.1, size=(self.n_users,)))
        if self.has_item_bias:
            self.item_bias = nn.Parameter(torch.normal(0, 0.1, size=(self.n_items,)))

        if self.config["has_user_emb"]:
            self.user_embedding = nn.Embedding(
                self.n_users, self.embedding_size, padding_idx=0
            )

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )  # if padding_idx is not set, the embedding vector of padding_idx will change during training
        if self.use_text_emb:
            # We load the pretrained text embedding, and freeze it during training.
            # But text_mlp is trainable to map the text embedding to the same space as item embedding.
            # Architecture of text_mlp is fixed as a simple 2-layer MLP.
            self.text_embedding = nn.Embedding(
                self.n_items, self.text_emb_size, padding_idx=0
            )
            self.text_embedding.weight.requires_grad_(False)
            self.text_mlp = nn.Sequential(
                nn.Linear(self.text_emb_size, 2 * self.hidden_size),
                nn.GELU(),
                nn.Linear(2 * self.hidden_size, self.hidden_size),
            )
        if self.use_features:
            # we merge all features into one embedding layer, for example, if we have 2 features, and each feature has 10 categories,
            # the feature1 index is from 0 to 9, and feature2 index is from 10 to 19.
            self.features_embedding = nn.Embedding(
                sum(self.features_shape), self.feature_emb_size, padding_idx=0
            )

        # model layers
        self._define_model_layers()
        self._init_params()

        if (
            self.use_pre_item_emb and "item_emb_path" in self.config
        ):  # check for item_emb_path is to ensure that in infer/test task (or other cases that need to load model ckpt), we don't load pre_item_emb which will be overwritten by model ckpt.
            pre_item_emb = file_io.load_pre_item_emb(
                self.config["item_emb_path"], self.logger
            )
            pre_item_emb = torch.from_numpy(pre_item_emb)
            pad_emb = torch.zeros([1, self.embedding_size])
            pre_item_emb = torch.cat([pad_emb, pre_item_emb], dim=-2).to(torch.float32)
            self.logger.info("{0}={1}".format("self.n_items", self.n_items))
            self.logger.info("{0}={1}".format("pre_item_emb", pre_item_emb))
            self.logger.info("{0}={1}".format("pre_item_emb.size", pre_item_emb.size()))
            self.item_embedding.weight = nn.Parameter(pre_item_emb)
        if self.use_text_emb and "text_emb_path" in self.config:
            text_emb = file_io.load_pre_item_emb(
                self.config["text_emb_path"], self.logger
            )
            text_emb = torch.from_numpy(text_emb)
            pad_emb = torch.zeros([1, self.text_emb_size])
            text_emb = torch.cat([pad_emb, text_emb], dim=-2).to(torch.float32)
            self.logger.info("{0}={1}".format("self.n_items", self.n_items))
            self.logger.info("{0}={1}".format("text_emb", text_emb))
            self.logger.info("{0}={1}".format("text_emb.size", text_emb.size()))
            self.text_embedding = nn.Embedding.from_pretrained(
                text_emb, freeze=True, padding_idx=0
            )

    def _init_params(self):
        init_methods = {
            "xavier_normal": xavier_normal_initialization,
            "xavier_uniform": xavier_uniform_initialization,
            "normal": normal_initialization(
                self.config["init_mean"], self.config["init_std"]
            ),
        }
        for name, module in self.named_children():
            init_method = init_methods[self.init_method]
            module.apply(init_method)

    def _cal_loss(self, scores, labels=None, reduction=True):
        r"""Calculate loss with scores and labels.

        Args:
            scores (torch.Tensor): scores of positive or negative items in batch.
            labels (torch.Tensor): labels of items. labels are not required in BPR and CCL loss since
                they are pairwise loss, in which the first item is positive and the residual is negative.
            reduction (bool): whether to reduce on the batch number dimension (usually the first dim).
                If True, a loss value would be returned. Otherwise, a tensor with shape (B,) would be returned.

        Return:
            loss ():
        """
        if self.group_size > 0:
            scores = scores.view(-1, self.group_size)
            if labels is not None:
                labels = labels.view(-1, self.group_size)

        ## trigger compliance validation
        if self.loss_type in [LossFuncType.BPR.value, LossFuncType.CCL.value]:
            if labels is not None and VALID_TRIGGER_P > random.random():
                has_pos = torch.gt(torch.sum(labels[:, 1:], dtype=torch.float32), 0.5)
                if has_pos.item():
                    raise ValueError(
                        r"""
                            For efficiency concern, we put the limitation in implementation:
                            if you want to use BPR or CCL as the loss function, please make sure only the first item in one group is positive item.  
                        """
                    )

        if self.loss_type == LossFuncType.BCE.value:
            logits = torch.clamp(nn.Sigmoid()(scores), min=-1 * EPS, max=1 - EPS)
            labels = labels.float()  # .to(self.device)
            loss = nn.BCELoss(reduction="mean" if reduction else "none")(
                logits, labels
            ).mean(dim=-1)
        elif self.loss_type == LossFuncType.BPR.value:
            neg_score = scores[
                :, 1:
            ]  ##-- currently only support one positive in item list.
            pos_score = scores[:, 0].unsqueeze(1).expand_as(neg_score)
            loss = modules.bpr_loss(pos_score, neg_score, reduction)
        elif self.loss_type == LossFuncType.CCL.value:
            neg_score = scores[
                :, 1:
            ]  ##-- currently only support one positive in item list.
            pos_score = scores[:, 0]
            loss = modules.ccl_loss(
                pos_score,
                neg_score,
                self.config["ccl_w"],
                self.config["ccl_m"],
                reduction,
            )
        elif self.loss_type == LossFuncType.SOFTMAX.value:
            scores = -nn.functional.log_softmax(
                scores, dim=-1
            )  # -torch.log(nn.Softmax(dim=-1)(scores) + EPS)
            # loss = scores[:, 0].mean()
            loss = scores[
                labels > 0
            ]  # The dimension of scores: [batch_size, group_size]
            if reduction:
                loss = loss.mean()
        elif self.loss_type == LossFuncType.FULLSOFTMAX.value:
            pos_scores = torch.gather(scores, 1, labels.reshape(-1, 1)).squeeze(-1)
            loss = torch.logsumexp(scores, dim=-1) - pos_scores
            if reduction:
                loss = loss.mean()

        return loss

    # def calculate_loss(self, interaction, reduction=True):
    #     item_id = interaction['item_id'] if self.loss_type != LossFuncType.FULLSOFTMAX.value else torch.arange(self.n_items).to(self.device)
    #     item_features = None
    #     if self.use_features:
    #         item_features = interaction['item_features'] if self.loss_type != LossFuncType.FULLSOFTMAX.value else torch.tensor(self.item2features, dtype=torch.int32).to(self.device)
    #     items_emb = self.forward_item_emb(item_id, item_features)
    #     user_emb = self.forward_user_emb(interaction)
    #     scores = self._predict_layer(user_emb, items_emb, interaction)
    #     labels = interaction['label'] if self.loss_type != LossFuncType.FULLSOFTMAX.value else interaction['item_id']
    #     loss = self._cal_loss(scores, labels, reduction)
    #     return loss

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        messages = []
        messages.append(super().__str__())
        messages.append("Trainable parameter number: {0}".format(params))

        messages.append("All trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                messages.append("{0} : {1}".format(name, param.size()))

        return "\n".join(messages)
