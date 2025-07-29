# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class RecoDataset(Dataset):
    """
    PyTorch Dataset for collaborative filtering tasks.

    Stores user, item, and rating data as tensors for efficient batching.
    """

    def __init__(self, users, items, ratings):
        """
        Args:
            users (array-like): User IDs or indices.
            items (array-like): Item IDs or indices.
            ratings (array-like): Ratings or interactions.
        """
        # Convert to numpy arrays first and ensure correct types
        users = np.array(users, dtype=np.int64)
        items = np.array(items, dtype=np.int64)
        ratings = np.array(ratings, dtype=np.float32)

        # Then convert to tensors
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of ratings.
        """
        return len(self.ratings)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (user_item_tensor, rating_tensor)
        """
        user_item_tensor = torch.stack((self.users[idx], self.items[idx]))
        rating_tensor = self.ratings[idx].unsqueeze(0)
        return user_item_tensor, rating_tensor


class RecoDataLoader:
    """
    Utility class for managing training and validation DataLoaders for collaborative filtering.

    Stores metadata about users/items and provides helper methods for data preparation and inspection.
    """

    def __init__(self, train_dl, valid_dl=None):
        """Initialize the dataloaders.

        Args:
            train_dl (DataLoader): Training dataloader
            valid_dl (DataLoader, optional): Validation dataloader
        """
        self.train = train_dl
        self.valid = valid_dl
        self.classes = {}

    @classmethod
    def from_df(
        cls,
        ratings,
        valid_pct=0.2,
        user_name=None,
        item_name=None,
        rating_name=None,
        seed=42,
        batch_size=64,
        **kwargs,
    ):
        """
        Create DataLoaders from a pandas DataFrame for collaborative filtering.

        Args:
            ratings (pd.DataFrame): DataFrame containing user, item, and rating columns.
            valid_pct (float): Fraction of data to use for validation.
            user_name (str): Name of the user column.
            item_name (str): Name of the item column.
            rating_name (str): Name of the rating column.
            seed (int): Random seed for reproducibility.
            batch_size (int): Batch size for DataLoaders.
            **kwargs: Additional DataLoader arguments.

        Returns:
            RecoDataLoader: Instance with train/valid DataLoaders and metadata.
        """
        # Validate input
        if ratings is None or len(ratings) == 0:
            raise ValueError("Input DataFrame is empty")

        # Set random seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Get column names
        user_name = user_name or ratings.columns[0]
        item_name = item_name or ratings.columns[1]
        rating_name = rating_name or ratings.columns[2]

        # Validate columns exist
        required_cols = [user_name, item_name, rating_name]
        if not all(col in ratings.columns for col in required_cols):
            raise ValueError(
                f"Missing required columns: {[col for col in required_cols if col not in ratings.columns]}"
            )

        # Drop any rows with NaN values
        ratings = ratings.dropna(subset=[user_name, item_name, rating_name])
        if len(ratings) == 0:
            raise ValueError("No valid data after dropping NaN values")

        # Get unique users and items (as strings)
        # Convert to string first to ensure consistent type for sorting
        users = ratings[user_name].astype(str).unique()
        items = ratings[item_name].astype(str).unique()

        if len(users) == 0 or len(items) == 0:
            raise ValueError("No unique users or items found in the data")

        # Sort unique users and items using standard string sorting
        # This matches the behavior observed in fastai's categorization for numeric strings
        sorted_users = ["#na#"] + sorted(users.tolist())
        sorted_items = ["#na#"] + sorted(items.tolist())

        # Create mapping dictionaries using the string-sorted lists
        user2idx = {u: i for i, u in enumerate(sorted_users)}
        item2idx = {i: idx for idx, i in enumerate(sorted_items)}

        # Convert original IDs in the DataFrame to indices using the mapping
        # Use .loc[] for assignment to avoid SettingWithCopyWarning
        ratings.loc[:, user_name] = (
            ratings[user_name]
            .astype(str)
            .map(user2idx)
            .fillna(user2idx["#na#"])
            .astype(np.int64)
        )
        ratings.loc[:, item_name] = (
            ratings[item_name]
            .astype(str)
            .map(item2idx)
            .fillna(item2idx["#na#"])
            .astype(np.int64)
        )
        ratings.loc[:, rating_name] = ratings[rating_name].astype(
            np.float32
        )  # Ensure rating is float

        # Split into train and validation
        n = len(ratings)
        n_valid = int(n * valid_pct)

        if n_valid >= n:
            if n == 0:
                raise ValueError(
                    "Input DataFrame was empty or contained no valid rows after cleaning."
                )
            else:
                raise ValueError(
                    f"Validation percentage {valid_pct} is too high. {n} total items, {n_valid} requested for validation leaves {n - n_valid} for training."
                )

        indices = list(range(n))
        random.shuffle(indices)
        train_idx = indices[n_valid:]
        valid_idx = indices[:n_valid]

        if len(train_idx) == 0:
            raise ValueError("Training set is empty after split. Reduce valid_pct.")

        # Create datasets using the index-mapped values
        train_ds = RecoDataset(
            ratings.iloc[train_idx][user_name].values,
            ratings.iloc[train_idx][item_name].values,
            ratings.iloc[train_idx][rating_name].values,
        )

        valid_ds = (
            RecoDataset(
                ratings.iloc[valid_idx][user_name].values,
                ratings.iloc[valid_idx][item_name].values,
                ratings.iloc[valid_idx][rating_name].values,
            )
            if n_valid > 0
            else None
        )

        # Create dataloaders with safe batch sizes
        train_dl = DataLoader(
            train_ds,
            batch_size=(
                min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
            ),  # Ensure batch_size isn't larger than dataset
            shuffle=True,
            **kwargs,
        )

        valid_batch_size = batch_size
        valid_dl = (
            DataLoader(
                valid_ds,
                batch_size=(
                    min(valid_batch_size, len(valid_ds))
                    if valid_ds and len(valid_ds) > 0
                    else (1 if valid_ds else None)
                ),
                shuffle=False,
                **kwargs,
            )
            if valid_ds is not None and len(valid_ds) > 0
            else None
        )  # Ensure valid_dl is None if valid_ds is empty

        # Create instance and store metadata
        dl = cls(train_dl, valid_dl)
        # Store the string-sorted lists in .classes
        dl.classes = {user_name: sorted_users, item_name: sorted_items}
        dl.user = user_name
        dl.item = item_name
        # n_users and n_items should be the size of the classes lists, including #na#
        dl.n_users = len(sorted_users)
        dl.n_items = len(sorted_items)
        dl.user2idx = user2idx  # Store mappings for potential later use
        dl.item2idx = item2idx  # Store mappings for potential later use

        return dl

    def show_batch(self, n=5):
        """
        Display a sample batch from the training DataLoader.

        Args:
            n (int): Number of examples to show from the batch.
        """
        # Get one batch from the training dataloader
        # Unpack the two elements from the batch: user_item_batch (tensor of shape [bs, 2]) and ratings_batch (tensor of shape [bs, 1])
        for user_item_batch, ratings_batch in self.train:
            batch_size = user_item_batch.shape[0]
            if n > batch_size:
                raise ValueError(
                    f"n ({n}) rows cannot be greater than the batch size ({batch_size})"
                )
            users = user_item_batch[:, 0]  # Shape [bs]
            items = user_item_batch[:, 1]  # Shape [bs]

            users = users[:n].numpy()
            items = items[:n].numpy()
            # Squeeze the ratings numpy array to remove the dimension of size 1
            ratings = ratings_batch[:n].numpy().squeeze()  # Shape [n]

            df = pd.DataFrame(
                {
                    self.user: [self.classes[self.user][u] for u in users],
                    self.item: [self.classes[self.item][i] for i in items],
                    "rating": ratings,
                }
            )

            print(f"Showing {n} examples from a batch:")
            print(df)
            break
