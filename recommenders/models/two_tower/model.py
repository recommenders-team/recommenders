# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F


class Tower(nn.Module):
    """One encoder of the Two-Tower model.

    The tower looks up a learned ID embedding for an entity (user or item),
    optionally concatenates fixed side features for that entity, and passes the
    result through an MLP. The output is L2-normalized so that the dot product
    between a user and an item vector is their cosine similarity.

    If there are no side features and no hidden layers, the tower reduces to a
    plain embedding lookup, i.e. the model becomes matrix factorization trained
    with a retrieval loss.
    """

    def __init__(
        self,
        n_entities,
        embedding_dim=64,
        hidden_units=(128,),
        output_dim=64,
        features=None,
        dropout=0.0,
        normalize=True,
    ):
        """Initialize the tower.

        Args:
            n_entities (int): Number of users or items (size of the ID vocabulary).
            embedding_dim (int): Size of the learned ID embedding.
            hidden_units (tuple): Sizes of the hidden layers of the MLP.
            output_dim (int): Size of the output vector. Ignored when there are
                no hidden layers and no features, in which case the output size
                is `embedding_dim`.
            features (torch.Tensor or numpy.ndarray): Optional dense side
                features of shape (n_entities, n_features). Row `i` holds the
                features of the entity with index `i`.
            dropout (float): Dropout probability applied after each hidden layer.
            normalize (bool): Whether to L2-normalize the output vectors.
        """
        super().__init__()
        if n_entities <= 0:
            raise ValueError("n_entities must be a positive integer")
        self.normalize = normalize
        self.embedding = nn.Embedding(n_entities, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.05)

        if features is not None:
            features = torch.as_tensor(features, dtype=torch.float32)
            if features.dim() != 2 or features.shape[0] != n_entities:
                raise ValueError(
                    f"features must have shape ({n_entities}, n_features), "
                    f"got {tuple(features.shape)}"
                )
            # A buffer moves with the model across devices and is saved in the
            # state dict, but it is not trained.
            self.register_buffer("features", features)
            feature_dim = features.shape[1]
        else:
            self.features = None
            feature_dim = 0

        hidden_units = tuple(hidden_units or ())
        if feature_dim == 0 and len(hidden_units) == 0:
            self.mlp = nn.Identity()
            self.output_dim = embedding_dim
        else:
            layers = []
            in_dim = embedding_dim + feature_dim
            for units in hidden_units:
                layers += [nn.Linear(in_dim, units), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = units
            layers.append(nn.Linear(in_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
            self.output_dim = output_dim

    def forward(self, idx):
        """Encode a batch of entities.

        Args:
            idx (torch.Tensor): Long tensor of entity indices, shape (batch_size,).

        Returns:
            torch.Tensor: Vectors of shape (batch_size, output_dim).
        """
        x = self.embedding(idx)
        if self.features is not None:
            x = torch.cat([x, self.features[idx]], dim=-1)
        x = self.mlp(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x


class TwoTowerModel(nn.Module):
    """Two-Tower (dual-encoder) model for candidate retrieval.

    Users and items are encoded independently by two towers into the same
    vector space, and the relevance of an item for a user is the dot product of
    their vectors. Because the item vectors do not depend on the user, they can
    be computed once and searched with (approximate) nearest-neighbour search,
    which makes the model suitable for retrieving candidates from large
    catalogues.

    References:
        Po-Sen Huang et al., "Learning Deep Structured Semantic Models for Web
        Search using Clickthrough Data", CIKM 2013.

        Xinyang Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large
        Corpus Item Recommendations", RecSys 2019.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        hidden_units=(128,),
        output_dim=64,
        user_features=None,
        item_features=None,
        dropout=0.0,
        normalize=True,
    ):
        """Initialize the model.

        Args:
            n_users (int): Number of users.
            n_items (int): Number of items.
            embedding_dim (int): Size of the learned ID embeddings in both towers.
            hidden_units (tuple): Hidden layer sizes of the MLP in both towers.
            output_dim (int): Size of the shared user/item vector space.
            user_features (torch.Tensor or numpy.ndarray): Optional user side
                features of shape (n_users, n_user_features).
            item_features (torch.Tensor or numpy.ndarray): Optional item side
                features of shape (n_items, n_item_features).
            dropout (float): Dropout probability in the MLPs.
            normalize (bool): Whether to L2-normalize the tower outputs.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_tower = Tower(
            n_users,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            features=user_features,
            dropout=dropout,
            normalize=normalize,
        )
        self.item_tower = Tower(
            n_items,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            output_dim=output_dim,
            features=item_features,
            dropout=dropout,
            normalize=normalize,
        )
        if self.user_tower.output_dim != self.item_tower.output_dim:
            raise ValueError(
                "User and item towers produce vectors of different sizes "
                f"({self.user_tower.output_dim} vs {self.item_tower.output_dim}). "
                "Use hidden layers or the same features setting in both towers."
            )

    def encode_users(self, user_idx):
        """Encode users into the shared vector space.

        Args:
            user_idx (torch.Tensor): Long tensor of user indices.

        Returns:
            torch.Tensor: User vectors of shape (len(user_idx), output_dim).
        """
        return self.user_tower(user_idx)

    def encode_items(self, item_idx=None):
        """Encode items into the shared vector space.

        Args:
            item_idx (torch.Tensor): Long tensor of item indices. If None, all
                items are encoded.

        Returns:
            torch.Tensor: Item vectors of shape (len(item_idx), output_dim).
        """
        if item_idx is None:
            device = self.item_tower.embedding.weight.device
            item_idx = torch.arange(self.n_items, device=device)
        return self.item_tower(item_idx)

    def forward(self, user_idx, item_idx):
        """Score user-item pairs.

        Args:
            user_idx (torch.Tensor): Long tensor of user indices, shape (batch_size,).
            item_idx (torch.Tensor): Long tensor of item indices, shape (batch_size,).

        Returns:
            torch.Tensor: Dot-product scores of shape (batch_size,).
        """
        return (self.encode_users(user_idx) * self.encode_items(item_idx)).sum(dim=-1)
