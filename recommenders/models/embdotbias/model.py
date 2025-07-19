# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import torch
from torch.nn import Embedding
from torch.nn import Module
import torch.nn.init as init


class EmbeddingDotBias(Module):
    "Base dot model for collaborative filtering."

    def __init__(self, n_factors, n_users, n_items, y_range=None):

        super().__init__()
        self.classes = None
        self.y_range = y_range
        self.u_weight = Embedding(n_users, n_factors)
        self.i_weight = Embedding(n_items, n_factors)
        self.u_bias = Embedding(n_users, 1)
        self.i_bias = Embedding(n_items, 1)

        # Initialize with truncated normal
        for emb in [self.u_weight, self.i_weight, self.u_bias, self.i_bias]:
            init.trunc_normal_(emb.weight, std=0.01)

    def forward(self, x):
        users, items = x[:, 0], x[:, 1]
        dot = self.u_weight(users) * self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.y_range is None:
            return res
        return (
            torch.sigmoid(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        )

    @classmethod
    def from_classes(cls, n_factors, classes, user=None, item=None, y_range=None):
        "Build a model with `n_factors` by inferring `n_users` and  `n_items` from `classes`"
        if user is None:
            user = list(classes.keys())[0]
        if item is None:
            item = list(classes.keys())[1]
        res = cls(n_factors, len(classes[user]), len(classes[item]), y_range=y_range)
        res.classes, res.user, res.item = classes, user, item
        return res

    def _get_idx(self, entity_ids, is_item=True):
        "Fetch item or user (based on `is_item`) for all in `entity_ids`"
        assert hasattr(
            self, "classes"
        ), "Build your model with `EmbeddingDotBias.from_classes` to use this functionality."
        classes = self.classes[self.item] if is_item else self.classes[self.user]
        # Create a mapping from entity ID (user or item) to its integer index in the embedding matrix
        entity_id_to_index = {entity_id: idx for idx, entity_id in enumerate(classes)}
        try:
            return torch.tensor([entity_id_to_index[o] for o in entity_ids])
        except KeyError as e:
            message = f"You're trying to access {'item' if is_item else 'user'} {entity_ids} that isn't in the training data. If it was in your original data, it may have been split such that it's only in the validation set now."
            raise KeyError(message)

    def bias(self, entity_ids, is_item=True):
        "Bias for item or user (based on `is_item`) for all in `entity_ids`"
        idx = self._get_idx(entity_ids, is_item)
        layer = (self.i_bias if is_item else self.u_bias).eval().cpu()
        return layer(idx).squeeze().detach()

    def weight(self, entity_ids, is_item=True):
        "Weight for item or user (based on `is_item`) for all in `entity_ids`"
        idx = self._get_idx(entity_ids, is_item)
        layer = (self.i_weight if is_item else self.u_weight).eval().cpu()
        return layer(idx).detach()
