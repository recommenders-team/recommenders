# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def in_batch_softmax_loss(
    user_vectors, item_vectors, item_idx, temperature=0.05, item_log_q=None
):
    """Sampled softmax loss using the other items in the batch as negatives.

    For the i-th (user, item) pair in the batch, the positive is item i and the
    negatives are all the other items in the batch. Rows where the same item
    appears more than once are masked so that a user's positive item is never
    used as one of its own negatives.

    Popular items show up as in-batch negatives more often than rare items,
    which biases the model against them. The logQ correction subtracts the log
    of each item's sampling probability from its logit to compensate (Yi et
    al., RecSys 2019).

    Args:
        user_vectors (torch.Tensor): User vectors, shape (batch_size, dim).
        item_vectors (torch.Tensor): Vectors of the positive items, shape (batch_size, dim).
        item_idx (torch.Tensor): Indices of the positive items, shape (batch_size,).
        temperature (float): Softmax temperature. Lower values sharpen the
            distribution; values around 0.05 work well with L2-normalized vectors.
        item_log_q (torch.Tensor): Log sampling probability of every item, shape
            (n_items,). If None, no correction is applied.

    Returns:
        torch.Tensor: Scalar loss.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = user_vectors @ item_vectors.T / temperature
    if item_log_q is not None:
        logits = logits - item_log_q[item_idx].unsqueeze(0)

    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)
    same_item = item_idx.unsqueeze(0) == item_idx.unsqueeze(1)
    accidental_hits = same_item & ~torch.eye(
        batch_size, dtype=torch.bool, device=logits.device
    )
    logits = logits.masked_fill(accidental_hits, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits, labels)


class Trainer:
    """Train a `TwoTowerModel` with the in-batch softmax loss."""

    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
        temperature=0.05,
        item_counts=None,
        device=None,
    ):
        """Initialize the trainer.

        Args:
            model (TwoTowerModel): Model to train.
            learning_rate (float): Learning rate of the Adam optimizer.
            weight_decay (float): Weight decay (L2 penalty) of the optimizer.
            temperature (float): Softmax temperature of the loss.
            item_counts (numpy.ndarray): Number of training interactions per
                item, shape (n_items,), e.g. `TwoTowerData.item_counts`. If
                given, the logQ correction is applied. If None, it is not.
            device (str or torch.device): Device to train on. Defaults to CUDA
                when available, otherwise CPU.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.item_log_q = None
        if item_counts is not None:
            item_counts = np.asarray(item_counts, dtype=np.float64)
            if len(item_counts) != model.n_items:
                raise ValueError(
                    f"item_counts has {len(item_counts)} entries, expected {model.n_items}"
                )
            # Items never seen in training can't be in a batch; clip avoids log(0)
            probs = np.clip(item_counts, 1, None) / item_counts.sum()
            self.item_log_q = torch.as_tensor(
                np.log(probs), dtype=torch.float32, device=self.device
            )

    def _batch_loss(self, user_idx, item_idx):
        user_idx = user_idx.to(self.device)
        item_idx = item_idx.to(self.device)
        user_vectors = self.model.encode_users(user_idx)
        item_vectors = self.model.encode_items(item_idx)
        return in_batch_softmax_loss(
            user_vectors,
            item_vectors,
            item_idx,
            temperature=self.temperature,
            item_log_q=self.item_log_q,
        )

    def train_epoch(self, train_dl):
        """Train the model for one epoch.

        Args:
            train_dl (torch.utils.data.DataLoader): Loader of (user, item) index pairs.

        Returns:
            float: Mean training loss over the epoch.
        """
        self.model.train()
        total_loss, n_batches = 0.0, 0
        for user_idx, item_idx in train_dl:
            if len(user_idx) < 2:
                # A single pair has no in-batch negatives
                continue
            self.optimizer.zero_grad()
            loss = self._batch_loss(user_idx, item_idx)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def validate(self, valid_dl):
        """Compute the mean loss on validation data without updating the model.

        Args:
            valid_dl (torch.utils.data.DataLoader): Loader of (user, item) index pairs.

        Returns:
            float: Mean validation loss.
        """
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for user_idx, item_idx in valid_dl:
                if len(user_idx) < 2:
                    continue
                total_loss += self._batch_loss(user_idx, item_idx).item()
                n_batches += 1
        return total_loss / max(n_batches, 1)

    def fit(self, train_dl, n_epochs, valid_dl=None, verbose=True):
        """Train the model.

        Args:
            train_dl (torch.utils.data.DataLoader): Training loader.
            n_epochs (int): Number of epochs.
            valid_dl (torch.utils.data.DataLoader): Optional validation loader.
            verbose (bool): Whether to log the losses after every epoch.

        Returns:
            list: One dict per epoch with keys `epoch`, `train_loss` and, if
            `valid_dl` is given, `valid_loss`.
        """
        history = []
        for epoch in range(1, n_epochs + 1):
            record = {"epoch": epoch, "train_loss": self.train_epoch(train_dl)}
            if valid_dl is not None:
                record["valid_loss"] = self.validate(valid_dl)
            history.append(record)
            if verbose:
                logger.info(
                    ", ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in record.items()
                    )
                )
        self.model.eval()
        return history
