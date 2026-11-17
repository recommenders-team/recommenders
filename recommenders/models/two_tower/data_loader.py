# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL

logger = logging.getLogger(__name__)


class InteractionDataset(Dataset):
    """PyTorch dataset of (user index, item index) positive pairs."""

    def __init__(self, user_idx, item_idx):
        """Initialize the dataset.

        Args:
            user_idx (numpy.ndarray): User indices.
            item_idx (numpy.ndarray): Item indices, aligned with `user_idx`.
        """
        if len(user_idx) != len(item_idx):
            raise ValueError("user_idx and item_idx must have the same length")
        # Copy so that later changes to the input arrays don't affect the dataset
        self.user_idx = torch.tensor(np.asarray(user_idx), dtype=torch.long)
        self.item_idx = torch.tensor(np.asarray(item_idx), dtype=torch.long)

    def __len__(self):
        """Number of interactions."""
        return len(self.user_idx)

    def __getitem__(self, idx):
        """Return the (user index, item index) pair at position `idx`."""
        return self.user_idx[idx], self.item_idx[idx]


class TwoTowerData:
    """Index users and items and prepare everything the Two-Tower model needs.

    It maps raw user/item IDs to contiguous indices, stores the training
    interactions as positive pairs, keeps item frequencies (used by the logQ
    correction of the in-batch softmax loss), builds the optional side feature
    matrices, and remembers which items each user has seen so they can be
    removed from recommendations.

    Every row of the training DataFrame is treated as a positive interaction.
    To use only some interactions (e.g. ratings >= 4), filter the DataFrame
    before passing it in.
    """

    def __init__(
        self,
        train,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        user_features=None,
        item_features=None,
    ):
        """Initialize the data object.

        Args:
            train (pandas.DataFrame): Training interactions with at least the
                user and item columns.
            col_user (str): User column name.
            col_item (str): Item column name.
            user_features (pandas.DataFrame): Optional user side features. It
                must contain `col_user` and one or more numeric feature columns.
                Users without a row get all-zero features.
            item_features (pandas.DataFrame): Optional item side features. It
                must contain `col_item` and one or more numeric feature columns.
                Items without a row get all-zero features.
        """
        if train is None or len(train) == 0:
            raise ValueError("Training DataFrame is empty")
        missing = [c for c in (col_user, col_item) if c not in train.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.col_user = col_user
        self.col_item = col_item

        train = train[[col_user, col_item]].dropna()
        if len(train) == 0:
            raise ValueError("No valid interactions after dropping NaN values")

        # Keep the order of first appearance so indices are deterministic
        self.users = pd.unique(train[col_user])
        self.items = pd.unique(train[col_item])
        self.user2idx = {u: i for i, u in enumerate(self.users)}
        self.item2idx = {it: i for i, it in enumerate(self.items)}
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.user_idx = train[col_user].map(self.user2idx).to_numpy(dtype=np.int64)
        self.item_idx = train[col_item].map(self.item2idx).to_numpy(dtype=np.int64)

        self.item_counts = np.bincount(self.item_idx, minlength=self.n_items)

        seen = csr_matrix(
            (
                np.ones(len(self.user_idx), dtype=np.float32),
                (self.user_idx, self.item_idx),
            ),
            shape=(self.n_users, self.n_items),
        )
        seen.sum_duplicates()
        self.seen = seen

        self.user_features = self._feature_matrix(
            user_features, col_user, self.user2idx, self.n_users, "user_features"
        )
        self.item_features = self._feature_matrix(
            item_features, col_item, self.item2idx, self.n_items, "item_features"
        )

    @staticmethod
    def _feature_matrix(features, col_id, id2idx, n_entities, name):
        """Build a dense (n_entities, n_features) float32 matrix aligned with the indices."""
        if features is None:
            return None
        if col_id not in features.columns:
            raise ValueError(f"{name} must contain the column '{col_id}'")
        feature_cols = [c for c in features.columns if c != col_id]
        if len(feature_cols) == 0:
            raise ValueError(f"{name} must contain at least one feature column")
        features = features.drop_duplicates(subset=col_id)
        values = features[feature_cols].to_numpy(dtype=np.float32)
        idx = features[col_id].map(id2idx)
        known = idx.notna().to_numpy()
        matrix = np.zeros((n_entities, len(feature_cols)), dtype=np.float32)
        matrix[idx[known].to_numpy(dtype=np.int64)] = values[known]
        n_missing = n_entities - int(known.sum())
        if n_missing > 0:
            logger.info(f"{n_missing} entities have no {name}; using zeros")
        return matrix

    def loader(self, df=None, batch_size=1024, shuffle=True, seed=None, **kwargs):
        """Create a DataLoader of (user index, item index) positive pairs.

        Args:
            df (pandas.DataFrame): Interactions to load. If None, the training
                interactions are used. Rows with users or items unknown to the
                training data are dropped.
            batch_size (int): Batch size. With the in-batch softmax loss, larger
                batches give more negatives per positive.
            shuffle (bool): Whether to shuffle the pairs every epoch.
            seed (int): Seed for the shuffling, for reproducibility.
            **kwargs: Extra arguments passed to `torch.utils.data.DataLoader`.

        Returns:
            torch.utils.data.DataLoader: The data loader.
        """
        if df is None:
            user_idx, item_idx = self.user_idx, self.item_idx
        else:
            user_idx = df[self.col_user].map(self.user2idx)
            item_idx = df[self.col_item].map(self.item2idx)
            known = (user_idx.notna() & item_idx.notna()).to_numpy()
            n_dropped = int((~known).sum())
            if n_dropped > 0:
                logger.info(f"Dropped {n_dropped} rows with unknown users or items")
            user_idx = user_idx[known].to_numpy(dtype=np.int64)
            item_idx = item_idx[known].to_numpy(dtype=np.int64)
            if len(user_idx) == 0:
                raise ValueError("No rows with known users and items")

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        dataset = InteractionDataset(user_idx, item_idx)
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            generator=generator,
            **kwargs,
        )
