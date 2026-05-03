# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from recommenders.evaluation.python_evaluation import ndcg_at_k


class _VAEModel(nn.Module):
    """PyTorch VAE neural network: encoder + reparameterization + decoder."""

    def __init__(
        self,
        original_dim: int,
        intermediate_dim: int,
        latent_dim: int,
        drop_encoder: float,
        drop_decoder: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_dropout = nn.Dropout(drop_encoder)
        self.encoder_h = nn.Linear(original_dim, intermediate_dim)
        self.z_mean_layer = nn.Linear(intermediate_dim, latent_dim)
        self.z_log_var_layer = nn.Linear(intermediate_dim, latent_dim)

        # Decoder
        self.decoder_h = nn.Linear(latent_dim, intermediate_dim)
        self.decoder_dropout = nn.Dropout(drop_decoder)
        self.decoder_out = nn.Linear(intermediate_dim, original_dim)

        # Match TF paper initializers: glorot_uniform weights, truncated_normal(std=0.001) biases
        for layer in [
            self.encoder_h,
            self.z_mean_layer,
            self.z_log_var_layer,
            self.decoder_h,
            self.decoder_out,
        ]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.trunc_normal_(layer.bias, std=0.001)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.encoder_h(self.encoder_dropout(x)))
        return self.z_mean_layer(h), self.z_log_var_layer(h)

    def reparameterize(
        self, z_mean: torch.Tensor, z_log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * z_log_var)
            eps = torch.randn_like(std)
            return z_mean + eps * std
        return z_mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.decoder_h(z))
        return torch.softmax(self.decoder_out(self.decoder_dropout(h)), dim=-1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_bar = self.decode(z)
        return x_bar, z_mean, z_log_var


class StandardVAE:
    """Standard Variational Autoencoders (VAE) for Collaborative Filtering implementation."""

    def __init__(
        self,
        n_users: int,
        original_dim: int,
        intermediate_dim: int = 200,
        latent_dim: int = 70,
        n_epochs: int = 400,
        batch_size: int = 100,
        k: int = 100,
        verbose: int = 1,
        drop_encoder: float = 0.5,
        drop_decoder: float = 0.5,
        beta: float = 1.0,
        annealing: bool = False,
        anneal_cap: float = 1.0,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Initialize class parameters.

        Args:
            n_users (int): Number of unique users in the train set.
            original_dim (int): Number of unique items in the train set.
            intermediate_dim (int): Dimension of intermediate space.
            latent_dim (int): Dimension of latent space.
            n_epochs (int): Number of epochs for training.
            batch_size (int): Batch size.
            k (int): number of top k items per user.
            verbose (int): Whether to show the training output or not.
            drop_encoder (float): Dropout percentage of the encoder.
            drop_decoder (float): Dropout percentage of the decoder.
            beta (float): a constant parameter β in the ELBO function,
                  when you are not using annealing (annealing=False)
            annealing (bool): option of using annealing method for training the model (True)
                  or not using annealing, keeping a constant beta (False)
            anneal_cap (float): maximum value that beta can take during annealing process.
            seed (int): Seed.
            save_path (str): Default path to save weights.
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.n_users = n_users
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.k = k
        self.verbose = verbose

        self.number_of_batches = self.n_users // self.batch_size

        self.anneal_cap = anneal_cap
        self.annealing = annealing
        self.beta: float = 0.0 if annealing else beta

        self.total_anneal_steps = (
            self.number_of_batches * (self.n_epochs - int(self.n_epochs * 0.2))
        ) // self.anneal_cap

        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._create_model()

    def _create_model(self) -> None:
        """Build model and optimizer."""
        self.model = _VAEModel(
            self.original_dim,
            self.intermediate_dim,
            self.latent_dim,
            self.drop_encoder,
            self.drop_decoder,
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001, eps=1e-7)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.2, patience=1, min_lr=0.0001
        )

    def _vae_loss(
        self,
        x: torch.Tensor,
        x_bar: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Calculate negative ELBO (NELBO)."""
        # Reconstruction error: sum over features, mean over batch
        # Matches TF: original_dim * binary_crossentropy averages over features then Keras averages over batch
        reconst_loss = torch.mean(
            torch.sum(F.binary_cross_entropy(x_bar, x, reduction="none"), dim=-1)
        )
        # Kullback–Leibler divergence
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        )
        return reconst_loss + beta * kl_loss

    def _iter_batches(self, x_train_gpu: torch.Tensor):
        """Yield shuffled mini-batches from a pre-loaded GPU tensor."""
        idx = torch.randperm(x_train_gpu.shape[0], device=self.device)
        for i in range(self.number_of_batches):
            batch_idx = idx[i * self.batch_size : (i + 1) * self.batch_size]
            yield x_train_gpu[batch_idx]

    def _score(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass in eval mode and return numpy score matrix."""
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            x_bar, _, _ = self.model(x_tensor)
        return x_bar.cpu().numpy()

    def fit(
        self,
        x_train: np.ndarray,
        x_valid: np.ndarray,
        x_val_tr: np.ndarray,
        x_val_te: np.ndarray,
        mapper,
    ) -> None:
        """Fit model with the train sets and validate on the validation set.

        Args:
            x_train (numpy.ndarray): The click matrix for the train set.
            x_valid (numpy.ndarray): The click matrix for the validation set.
            x_val_tr (numpy.ndarray): The click matrix for the validation set training part.
            x_val_te (numpy.ndarray): The click matrix for the validation set testing part.
            mapper (object): The mapper for converting click matrix to dataframe. It can be AffinityMatrix.
        """
        self.model.to(self.device)

        # Preload datasets to GPU once to avoid repeated CPU→GPU transfers
        x_train_gpu = torch.FloatTensor(x_train).to(self.device)
        x_val_gpu = torch.FloatTensor(x_valid).to(self.device)

        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.val_ndcg: List[float] = []
        self.ls_beta: List[float] = []

        best_ndcg = 0.0
        update_count = 0

        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for batch in self._iter_batches(x_train_gpu):
                if self.annealing:
                    update_count += 1
                    self.beta = min(
                        update_count / self.total_anneal_steps, self.anneal_cap
                    )

                self.optimizer.zero_grad()
                x_bar, z_mean, z_log_var = self.model(batch)
                loss = self._vae_loss(batch, x_bar, z_mean, z_log_var, self.beta)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / self.number_of_batches

            # Validation loss (already on GPU)
            self.model.eval()
            with torch.no_grad():
                x_bar_val, z_mean_val, z_log_var_val = self.model(x_val_gpu)
                val_loss = self._vae_loss(
                    x_val_gpu, x_bar_val, z_mean_val, z_log_var_val, self.beta
                ).item()

            self.scheduler.step(val_loss)
            self.train_loss.append(avg_train_loss)
            self.val_loss.append(val_loss)
            self.ls_beta.append(self.beta)

            # NDCG@k on validation
            top_k = self._recommend_k_items_internal(x_val_tr, self.k, remove_seen=True)
            top_k_df = mapper.map_back_sparse(top_k, kind="prediction")
            test_df = mapper.map_back_sparse(x_val_te, kind="ratings")
            ndcg = ndcg_at_k(test_df, top_k_df, col_prediction="prediction", k=self.k)
            self.val_ndcg.append(ndcg)

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                if self.save_path is not None:
                    torch.save(self.model.state_dict(), self.save_path)

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} — "
                    f"loss: {avg_train_loss:.4f}  val_loss: {val_loss:.4f}  "
                    f"NDCG@{self.k}: {ndcg:.4f}"
                )

    @property
    def optimal_beta(self) -> float:
        """Returns the value of the optimal beta."""
        index_max_ndcg = np.argmax(self.val_ndcg)
        return self.ls_beta[index_max_ndcg]

    def display_metrics(self) -> None:
        """Plots loss and NDCG@k per epoch for train and validation sets."""
        plt.figure(figsize=(14, 5))
        sns.set(style="whitegrid")

        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss, color="b", linestyle="-", label="Train")
        plt.plot(self.val_loss, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("Loss", size=14)
        plt.legend(loc="upper left")

        plt.subplot(1, 2, 2)
        plt.plot(self.val_ndcg, color="r", linestyle="-", label="Val")
        plt.title("\n")
        plt.xlabel("Epochs", size=14)
        plt.ylabel("NDCG@k", size=14)
        plt.legend(loc="upper left")

        plt.suptitle("TRAINING AND VALIDATION METRICS HISTORY", size=16)
        plt.tight_layout(pad=2)

    def _recommend_k_items_internal(
        self, x: np.ndarray, k: int, remove_seen: bool = True
    ) -> np.ndarray:
        """Run inference and return top-k sparse score matrix."""
        score = self._score(x)
        if remove_seen:
            seen_mask = np.not_equal(x, 0)
            score[seen_mask] = 0
        top_items = np.argpartition(-score, range(k), axis=1)[:, :k]
        score_c = score.copy()
        score_c[np.arange(score_c.shape[0])[:, None], top_items] = 0
        return score - score_c

    def recommend_k_items(
        self, x: np.ndarray, k: int, remove_seen: bool = True
    ) -> np.ndarray:
        """Returns the top-k items ordered by a relevancy score.

        Obtained probabilities are used as recommendation score.

        Args:
            x (numpy.ndarray): Input click matrix, with `int32` values.
            k (scalar): The number of items to recommend.

        Returns:
            numpy.ndarray: A sparse matrix containing the top_k elements ordered by their score.
        """
        if self.save_path is not None:
            self.model.load_state_dict(
                torch.load(self.save_path, map_location=self.device, weights_only=True)
            )
        # Run inference on CPU to free GPU memory
        train_device = self.device
        self.device = torch.device("cpu")
        self.model.to(self.device)
        result = self._recommend_k_items_internal(x, k, remove_seen)
        self.device = train_device
        self.model.to(self.device)
        return result

    @property
    def ndcg_per_epoch(self) -> List[float]:
        """Returns the list of NDCG@k at each epoch."""
        return self.val_ndcg
