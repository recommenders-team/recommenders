# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn

from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL
import recommenders.utils.python_utils as pu
import recommenders.utils.torch_utils as tu

@dataclass(kw_only=True, frozen=True)
class WideAndDeepHyperParams:
    user_dim: int = 32
    item_dim: int = 32
    crossed_feat_dim: int = 1000
    dnn_hidden_units: Tuple[int, ...] = (128, 128)
    dnn_dropout: float = 0.0
    dnn_additional_embeddings_sizes: dict[str, Tuple[int, int]] = field(default_factory=dict)
    dnn_cont_features: int = 0

class WideAndDeepModel(nn.Module):
    def __init__(
        self, 
        num_users: int, 
        num_items: int, 
        hparams: WideAndDeepHyperParams = WideAndDeepHyperParams(),
        binary_output: bool = False,
    ):
        super().__init__()

        self.hparams = hparams
        self.n_users = num_users
        self.n_items = num_items
        
        self.users_emb = nn.Embedding(num_users, hparams.user_dim)
        self.items_emb = nn.Embedding(num_items, hparams.item_dim)
        self.additional_embs = nn.ModuleDict({
            k: nn.Embedding(num, dim) for k, (num, dim) in hparams.dnn_additional_embeddings_sizes.items()
        })

        # Randomly initialize embeddings
        total_emb_dim = hparams.user_dim + hparams.item_dim
        for _, emb in self.additional_embs.items():
            total_emb_dim += emb.embedding_dim
            nn.init.uniform_(emb.weight, -1, 1)

        layers = []
        prev_output = hparams.dnn_cont_features + total_emb_dim
        for hu in hparams.dnn_hidden_units:
            layers.append(nn.Linear(prev_output, hu))
            layers.append(nn.Dropout(hparams.dnn_dropout))
            layers.append(nn.ReLU())
            prev_output = hu

        self.deep = nn.Sequential(*layers)

        # Cross product of users-items
        exclusive_wide_input = hparams.crossed_feat_dim

        self.head = nn.Sequential(
            nn.Linear(exclusive_wide_input+prev_output, 1),
        )

        if binary_output:
            self.head.append(nn.Sigmoid())

    def forward(
        self, 
        interactions: torch.Tensor, 
        additional_embeddings: Dict[str, torch.Tensor] = {},
        continuous_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        users, items = interactions.T
    
        all_embed = torch.cat([
            self.users_emb(users), # Receives the indices
            self.items_emb(items),
            *[ emb(additional_embeddings[k]) for k, emb in self.additional_embs.items() ]
        ], dim=1)

        # TODO: Use hashing to avoid problems with biased distributions
        cross_product_idx = (users*self.n_items + items) % self.hparams.crossed_feat_dim
        cross_product = nn.functional.one_hot(cross_product_idx, self.hparams.crossed_feat_dim)

        if self.hparams.dnn_cont_features > 0:
            deep_input = torch.cat([continuous_features, all_embed], dim=1)
        else:
            deep_input = all_embed

        return self.head(torch.cat([
            cross_product, # wide input
            self.deep(deep_input), # deep output
        ], dim=1))


class WideAndDeepDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        ratings: pd.DataFrame, 
        user_col: str = DEFAULT_USER_COL, 
        item_col: str = DEFAULT_ITEM_COL,
        rating_col: str = DEFAULT_RATING_COL,
        n_users: Optional[int] = None, 
        n_items: Optional[int] = None,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.ratings = ratings.copy()

        self.n_users = n_users or ratings[user_col].max()+1
        self.n_items = n_items or ratings[item_col].max()+1

        self.ratings[rating_col] = self.ratings[rating_col].astype('float32')
    
    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # TODO: Get not only the interactions, but the continuous features and add. embs too
        item = self.ratings.iloc[idx]
        return { 
            'interactions': self.ratings[[self.user_col, self.item_col]].iloc[idx].values,
        }, item[self.rating_col]
    

class WideAndDeep(object):
    def __init__(
        self, 
        train: WideAndDeepDataset,
        test: WideAndDeepDataset,
        hparams: WideAndDeepHyperParams = WideAndDeepHyperParams(),
        *,
        n_users: Optional[int] = None,
        n_items: Optional[int] = None,
        epochs: int = 100,
        batch_size: int = 128,
        loss_fn: str | nn.Module = 'mse',
        optimizer: str = 'sgd',
        l1: float = 0.0001,
        optimizer_params: dict[str, Any] = dict(),
        disable_batch_progress: bool = False,
        disable_iter_progress: bool = False,
        prediction_col: str = DEFAULT_PREDICTION_COL,
    ):
        self.n_users = n_users or max(train.n_users, test.n_users)
        self.n_items = n_items or max(train.n_items, test.n_items)
        
        self.model = WideAndDeepModel(
            num_users=self.n_users,
            num_items=self.n_items,
            hparams=hparams,
        )

        self.train = train
        self.test = test
        self.train_dataloader = DataLoader(train, batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test, len(test))
        
        if isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = tu.LOSS_DICT[loss_fn]()
        
        self.optimizer = tu.OPTIM_DICT[optimizer](
            self.model.parameters(), 
            lr=l1,
            **optimizer_params,
        )

        self.disable_batch_progress = disable_batch_progress
        self.disable_iter_progress = disable_iter_progress
        self.prediction_col = prediction_col

        self.current_epoch = 0
        self.epochs = epochs

        self.train_loss_history = list()
        self.test_loss_history = list()

    @property
    def user_col(self) -> str:
        return self.train.user_col

    @property
    def item_col(self) -> str:
        return self.train.item_col
        
    def fit(self):
        if self.current_epoch >= self.epochs:
            print(f"Model is already trained with {self.epochs} epochs. Increment the number of epochs.")
        
        with tqdm(total=self.epochs, leave=True, disable=self.disable_iter_progress) as pbar:
            pbar.update(self.current_epoch)
            for _ in range(self.current_epoch, self.epochs):
                self.fit_step()
                pbar.update()
                pbar.set_postfix(
                    train_loss=self.train_loss_history[-1],
                    test_loss=self.test_loss_history[-1],
                )

    def fit_step(self):
        self.model.train()
        
        train_loss = 0.0
        for X,y in tqdm(self.train_dataloader, 'batch', leave=False, disable=self.disable_batch_progress):
            pred = self.model(X['interactions'])
            loss = self.loss_fn(pred, y)
            # TODO: Can we use this loss? Or should I calculate it again with no_grad?
            train_loss += loss.item()

            # Propagate error
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.train_loss_history.append(train_loss / len(self.train_dataloader))
        self.model.eval()
        
        num_batches = len(self.test_dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X['interactions'])
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        self.test_loss_history.append(test_loss)
    
        self.current_epoch += 1

    def recommend_k_items(
        self, user_ids=None, item_ids=None, top_k=10, remove_seen=True,
    ):
        if user_ids is None:
            user_ids = np.arange(1, self.n_users)
        if item_ids is None:
            item_ids = np.arange(1, self.n_items)

        uip = pd.MultiIndex.from_product(
            [user_ids, item_ids], 
            names=[self.user_col, self.item_col],
        )

        if remove_seen:
            uip = uip.difference(
                self.train.ratings.set_index([self.user_col, self.item_col]).index
            )
        
        uip = uip.to_frame(index=False)
        
        with torch.no_grad():
            uip[self.prediction_col] = self.model(torch.from_numpy(uip[[self.user_col, self.item_col]].values))

        return (
            uip
            .sort_values([self.user_col, self.prediction_col], ascending=[True, False])
            .groupby(self.user_col)
            .head(top_k)
            .reset_index(drop=True)
        )