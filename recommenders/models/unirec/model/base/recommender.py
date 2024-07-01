# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/model/base/recommender.py
#

from typing import *
import numpy as np
import inspect
import torch
import torch.nn as nn

import recommenders.models.unirec.model.modules as modules
from recommenders.models.unirec.constants.loss_funcs import LossFuncType
from recommenders.models.unirec.model.base.reco_abc import AbstractRecommender


class BaseRecommender(AbstractRecommender):

    def _init_attributes(self):
        super(BaseRecommender, self)._init_attributes()
        config = self.config
        self.dnn_inner_size = self.embedding_size
        self.time_seq = config.get("time_seq", 0)

    def _init_modules(self):
        # predict_layer
        scorer_type = self.config["distance_type"]
        if scorer_type == "mlp":
            self.scorer_layers = modules.MLPScorer(
                self.embedding_size,
                self.dnn_inner_size,
                self.dropout_prob,
                act_f="tanh",
            ).to(self.device)
        elif scorer_type == "dot":
            self.scorer_layers = modules.InnerProductScorer().to(self.device)
        elif scorer_type == "cosine":
            self.scorer_layers = modules.CosineScorer(eps=1e-6).to(self.device)
        else:
            raise ValueError("not supported distance_type: {0}".format(scorer_type))

        if self.time_seq:
            self.time_embedding = nn.Embedding(
                self.time_seq, self.embedding_size, padding_idx=0
            )

        super(BaseRecommender, self)._init_modules()

    def _define_model_layers(self):
        pass

    def forward_user_emb(
        self,
        user_id=None,
        item_seq=None,
        item_seq_len=None,
        item_seq_features=None,
        time_seq=None,
    ):
        user_e = self.user_embedding(user_id)
        return user_e

    def forward(
        self,
        user_id=None,
        item_id=None,
        label=None,
        item_features=None,
        item_seq=None,
        item_seq_len=None,
        item_seq_features=None,
        time_seq=None,
        session_id=None,
        reduction=True,
        return_loss_only=True,
    ):
        if self.loss_type == LossFuncType.FULLSOFTMAX.value and self.training:
            label = item_id
            in_item_id = torch.arange(self.n_items).to(self.device)
            in_item_features = (
                torch.tensor(self.item2features, dtype=torch.int32).to(self.device)
                if self.use_features
                else None
            )
        else:
            in_item_id = item_id
            in_item_features = item_features

        items_emb = self.forward_item_emb(in_item_id, in_item_features)
        user_emb = self.forward_user_emb(
            user_id, item_seq, item_seq_len, item_seq_features, time_seq
        )
        scores = self._predict_layer(user_emb, items_emb, user_id, in_item_id)
        if self.training:
            loss = self._cal_loss(scores, label, reduction)
            if return_loss_only:
                return loss, None, None, None
            return loss, scores, user_emb, items_emb
        else:
            return None, scores, user_emb, items_emb

    def forward_item_emb(self, items, item_features=None):
        item_emb = self.item_embedding(
            items
        )  # [batch_size, n_items_inline, embedding_size]
        if self.use_features:
            item_features_emb = self.features_embedding(item_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(items))
            item_emb = item_emb + text_emb
        return item_emb

    def _predict_layer(self, user_emb, items_emb, user_id, item_id):
        scores = self.scorer_layers(user_emb, items_emb)

        if self.has_user_bias:
            user_bias = self.user_bias[user_id]
            if scores.shape != user_bias.shape:
                user_bias = self.user_bias[user_id].unsqueeze(1)
                user_bias = torch.repeat_interleave(user_bias, scores.shape[-1], dim=-1)
            scores = scores + user_bias

        if self.has_item_bias:
            item_bias = self.item_bias[item_id]
            scores = scores + item_bias

        scores = scores / self.tau

        if self.SCORE_CLIP > 0:
            scores = torch.clamp(
                scores, min=-1.0 * self.SCORE_CLIP, max=self.SCORE_CLIP
            )
        return scores

    def predict(self, interaction):
        items_emb = self.forward_item_emb(
            interaction["item_id"],
            interaction["item_features"] if self.use_features else None,
        )
        inputs = {
            k: v
            for k, v in interaction.items()
            if k in inspect.signature(self.forward_user_emb).parameters
        }
        user_emb = self.forward_user_emb(**inputs)
        user_id = interaction["user_id"] if "user_id" in interaction else None
        item_id = interaction["item_id"] if "item_id" in interaction else None
        scores = (
            self._predict_layer(user_emb, items_emb, user_id, item_id)
            .detach()
            .cpu()
            .numpy()
        )
        return scores

    def forward_all_item_emb(self, batch_size=None, numpy=True):
        ### get all item's embeddings. when batch_size=None, it will proceed all in one run.
        ### when numpy=False, it would return a torch.Tensor
        if numpy:
            res = np.zeros((self.n_items, self.embedding_size), dtype=np.float32)
        else:
            res = torch.zeros(
                (self.n_items, self.embedding_size),
                dtype=torch.float32,
                device=self.device,
            )
        if batch_size is None:
            batch_size = self.n_items

        n_batch = (self.n_items - 1) // batch_size + 1
        for batch in range(n_batch):
            start = batch * batch_size
            end = min(self.n_items, start + batch_size)
            cur_items = torch.arange(start, end, dtype=torch.int32, device=self.device)
            item_features = (
                torch.tensor(self.item2features[start:end], dtype=torch.int32).to(
                    self.device
                )
                if self.use_features
                else None
            )
            cur_items_emb = self.forward_item_emb(cur_items, item_features).detach()
            if numpy:
                cur_items_emb = cur_items_emb.cpu().numpy()
            res[start:end] = cur_items_emb
        return res

    def get_all_item_bias(self):
        return self.item_bias.detach().cpu().numpy()

    def get_user_bias(self, interaction):
        return self.user_bias[interaction["user_id"]].detach().cpu().numpy()

    def item_embedding_for_user(self, item_seq, item_seq_features=None, time_seq=None):
        item_emb = self.item_embedding(item_seq)
        if self.use_features:
            item_features_emb = self.features_embedding(item_seq_features).sum(-2)
            item_emb = item_emb + item_features_emb
        if self.time_seq:
            time_embedding = self.time_embedding(time_seq)
            item_emb = item_emb + time_embedding
        if self.use_text_emb:
            text_emb = self.text_mlp(self.text_embedding(item_seq))
            item_emb = item_emb + text_emb
        return item_emb

    def topk(
        self,
        interaction: Dict[str, torch.Tensor],
        k: int,
        user_hist: torch.Tensor = None,
        candidates: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return topk items for a batch of users

        Compared with two-stage logic: score and get topk, the method is more simple to use. And the topk operation is done on GPU, which is
        faster then using numpy. A weakness is that user history should be padded to the same length, but it's not required to put on GPU. Now
        the method is used in MoRec Data Sampler, which is used to gather topk items for alignment objective.

        Args:
            interaction (Dict[str, torch.Tensor]): information of the user, including user id, item seq and other required information in `forward_user_emb`.
            k (int): top-k number.
            user_hist (torch.Tensor): padded batchified user history.
            candidates (torch.Tensor): item candidates id. When it's None, regard all items as candidates. The shape of candidates should be [batch_size, #candidates].

        Returns:
            (torch.Tensor, torch.Tensor): (top-k scores, top-k item ids).

        Note: user_hist should be padded to the max length of users' histories in the batch but not the max length of histories of all users, which
        could save memory and improve the efficiency.
        """
        inputs = {
            k: v
            for k, v in interaction.items()
            if k in inspect.signature(self.forward_user_emb).parameters
        }
        user_emb = self.forward_user_emb(**inputs)
        all_item_emb = self.forward_all_item_emb(numpy=False)
        user_id = interaction["user_id"] if "user_id" in interaction else None
        if candidates is None:
            candidates = torch.arange(
                len(all_item_emb), dtype=torch.long, device=all_item_emb.device
            )
            __all_item = True
        else:
            __all_item = False

        all_scores = self._predict_layer(user_emb, all_item_emb, user_id, candidates)

        if user_hist is not None:
            # Mask items in user history
            if __all_item:
                row_idx = (
                    torch.arange(user_emb.size(0), dtype=torch.long)
                    .unsqueeze_(-1)
                    .expand_as(user_hist)
                )
                all_scores[row_idx, user_hist] = -torch.inf
            else:
                sorted_hist, indices = torch.sort(user_hist, dim=-1)
                _idx = torch.searchsorted(sorted_hist, candidates, side="left")
                _idx[_idx == sorted_hist.size(-1)] = sorted_hist.size(-1) - 1
                _eq = torch.gather(sorted_hist, -1, _idx) == candidates
                all_scores[_eq] = -torch.inf

        topk_scores, _ids = torch.topk(all_scores, k, dim=-1)
        if __all_item:
            topk_ids = _ids
        else:
            topk_ids = torch.gather(candidates, -1, _ids)
        return topk_scores, topk_ids
