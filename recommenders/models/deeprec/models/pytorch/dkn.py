# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""DKN (Deep Knowledge-Aware Network) for news recommendation.

The knowledge-aware CNN (KCNN) embeds a news title. Every word position stacks
three channels: the word embedding, the embedding of the knowledge-graph entity
the word links to, and the embedding of that entity's context. A bank of
convolutions with max-over-time pooling turns them into one vector.

DKN embeds the candidate article and every clicked article with the same KCNN,
pools the clicked ones into a user vector with an attention network conditioned on
the candidate, and scores the (user, candidate) pair with a fully-connected head.

:Citation:

    H. Wang, F. Zhang, X. Xie and M. Guo, "DKN: Deep Knowledge-Aware Network for
    News Recommendation", in Proceedings of the 2018 World Wide Web Conference on
    World Wide Web, 2018.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.models.deeprec.deeprec_utils import cal_metric
from recommenders.models.deeprec.io.dkn_dataset import DKNDataset
from recommenders.models.deeprec.models.pytorch.fcn_net import FcnNet

__all__ = ["KCNN", "DKNBase", "DKN"]


def glorot_truncated_normal_(tensor: torch.Tensor, fan_avg: float) -> None:
    """Glorot normal init, truncated at two standard deviations.

    The standard deviation is rescaled so that the truncated distribution keeps
    the Glorot variance ``1 / fan_avg``.
    """
    std = math.sqrt(1.0 / fan_avg) / 0.87962566103423978
    nn.init.trunc_normal_(tensor, std=std, a=-2 * std, b=2 * std)


def init_weight_(tensor: torch.Tensor, init_method: str, init_value: float) -> None:
    """Initialize a weight or a bias of the attention, scorer or KCNN layers.

    Args:
        init_method (str): ``uniform`` draws from ``[-init_value, init_value]``;
            ``xavier_normal`` is :func:`glorot_truncated_normal_`, for which a
            bias of size ``n`` counts ``n`` as both fans.
    """
    if init_method == "uniform":
        nn.init.uniform_(tensor, -init_value, init_value)
    elif init_method == "xavier_normal":
        fan_avg = tensor.shape[0] if tensor.dim() == 1 else sum(tensor.shape) / 2
        glorot_truncated_normal_(tensor, fan_avg)
    else:
        raise ValueError(
            "init_method must be uniform or xavier_normal, but now is {0}".format(
                init_method
            )
        )


def _knowledge_transform(knowledge_dim: int, dim: int) -> nn.Linear:
    """``tanh`` layer mapping fixed entity or context embeddings to the word size."""
    transform = nn.Linear(knowledge_dim, dim)
    nn.init.uniform_(transform.weight, -1.0, 1.0)
    nn.init.zeros_(transform.bias)
    return transform


class KCNN(nn.Module):
    """Knowledge-aware CNN that embeds a news title."""

    def __init__(
        self,
        word_embeddings: np.ndarray,
        entity_embeddings: np.ndarray | None,
        context_embeddings: np.ndarray | None,
        filter_sizes: list[int],
        num_filters: int,
        init_method: str,
        init_value: float,
    ) -> None:
        """Initialize parameters.

        Args:
            word_embeddings (numpy.ndarray): Pre-trained ``[word_count, dim]`` word
                embeddings, fine-tuned during training.
            entity_embeddings (numpy.ndarray): Pre-trained
                ``[entity_count, entity_dim]`` entity embeddings. They stay fixed
                and a trainable ``tanh`` layer maps them to ``dim``. ``None``
                leaves the entity channel out.
            context_embeddings (numpy.ndarray): Entity context embeddings, looked
                up by entity index and handled like ``entity_embeddings``. ``None``
                leaves the context channel out.
            filter_sizes (list[int]): Window size of each convolution.
            num_filters (int): Filters per window size.
            init_method (str): Convolution bias init, see :func:`init_weight_`.
            init_value (float): Range of the ``uniform`` init.
        """
        super().__init__()
        if context_embeddings is not None and entity_embeddings is None:
            raise ValueError(
                "context_embeddings requires entity_embeddings: contexts are "
                "looked up by entity index."
            )
        self.use_entity = entity_embeddings is not None
        self.use_context = context_embeddings is not None

        self.word_embedding = nn.Parameter(
            torch.tensor(word_embeddings, dtype=torch.float32)
        )
        dim = self.word_embedding.shape[1]
        if self.use_entity:
            self.register_buffer(
                "entity_embedding", torch.tensor(entity_embeddings, dtype=torch.float32)
            )
            self.entity_transform = _knowledge_transform(
                entity_embeddings.shape[1], dim
            )
        if self.use_context:
            self.register_buffer(
                "context_embedding",
                torch.tensor(context_embeddings, dtype=torch.float32),
            )
            self.context_transform = _knowledge_transform(
                context_embeddings.shape[1], dim
            )

        channels = dim * (1 + self.use_entity + self.use_context)
        self.convs = nn.ModuleList()
        for size in filter_sizes:
            conv = nn.Conv1d(channels, num_filters, size)
            # The filter is a [size, channels] window over a one-channel
            # [doc_size, channels] image, so the channels count towards both fans.
            glorot_truncated_normal_(
                conv.weight, size * channels * (1 + num_filters) / 2
            )
            init_weight_(conv.bias, init_method, init_value)
            self.convs.append(conv)
        self.output_dim = num_filters * len(filter_sizes)

    def embedding_tables(self) -> list[torch.Tensor]:
        """The word table, then the transformed entity and context tables."""
        tables = [self.word_embedding]
        if self.use_entity:
            tables.append(torch.tanh(self.entity_transform(self.entity_embedding)))
        if self.use_context:
            tables.append(torch.tanh(self.context_transform(self.context_embedding)))
        return tables

    def forward(self, words: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        """words, entities ``[N, doc_size]`` -> title embeddings ``[N, output_dim]``."""
        word_table, *knowledge_tables = self.embedding_tables()
        channels = [F.embedding(words, word_table)] + [
            F.embedding(entities, table) for table in knowledge_tables
        ]
        x = torch.cat(channels, dim=-1).transpose(1, 2)
        return torch.cat([F.relu(conv(x)).amax(dim=-1) for conv in self.convs], dim=1)


class DKNBase(nn.Module):
    """KCNN plus the training lifecycle shared by DKN and DKNItem2Item.

    A subclass sets ``iterator`` and appends its own weights to ``layer_params``,
    and implements ``forward``, ``run_eval``, ``_data_loss(batch)`` and
    ``_embed_news(words, entities)``.
    """

    def __init__(
        self,
        word_embedding_file: str,
        entity_embedding_file: str | None,
        context_embedding_file: str | None,
        filter_sizes: list[int] | None,
        num_filters: int,
        init_method: str,
        init_value: float,
        seed: int | None,
    ) -> None:
        """Build the KCNN; see :class:`DKN` for the arguments."""
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.kcnn = KCNN(
            np.load(word_embedding_file),
            None if entity_embedding_file is None else np.load(entity_embedding_file),
            None if context_embedding_file is None else np.load(context_embedding_file),
            filter_sizes if filter_sizes is not None else [1, 2, 3],
            num_filters,
            init_method,
            init_value,
        )
        # The weights and biases the layer regularization applies to.
        self.layer_params = list(self.kcnn.convs.parameters())

    @property
    def device(self) -> torch.device:
        """Device the model's parameters live on."""
        return self.kcnn.word_embedding.device

    def _to_tensors(self, np_batch: dict) -> dict:
        """Move a batch of loader arrays onto the model's device, dtypes intact."""
        device = self.device
        return {
            key: torch.as_tensor(value, device=device)
            for key, value in np_batch.items()
        }

    def _regular_loss(
        self, embed_l2: float, embed_l1: float, layer_l2: float, layer_l1: float
    ) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        if embed_l2 > 0 or embed_l1 > 0:
            for table in self.kcnn.embedding_tables():
                if embed_l2 > 0:
                    reg = reg + embed_l2 * 0.5 * table.pow(2).sum()
                if embed_l1 > 0:
                    reg = reg + embed_l1 * table.abs().sum()
        for param in self.layer_params:
            if layer_l2 > 0:
                reg = reg + layer_l2 * 0.5 * param.pow(2).sum()
            if layer_l1 > 0:
                reg = reg + layer_l1 * param.abs().sum()
        return reg

    def fit(
        self,
        train_file: str,
        valid_file: str,
        test_file: str | None = None,
        epochs: int = 10,
        batch_size: int = 100,
        learning_rate: float = 0.0005,
        embed_l2: float = 1e-6,
        embed_l1: float = 0.0,
        layer_l2: float = 1e-6,
        layer_l1: float = 0.0,
        max_grad_norm: float | None = None,
        metrics: list[str] | None = None,
        pairwise_metrics: list[str] | None = None,
        show_step: int = 10000,
        model_dir: str | None = None,
        save_epoch: int = 2,
    ):
        """Train on ``train_file``, evaluating on ``valid_file`` after every epoch.

        Args:
            train_file (str): Training data set.
            valid_file (str): Validation set, evaluated every epoch.
            test_file (str): Optional test set, also evaluated every epoch.
            epochs (int): Number of training epochs.
            batch_size (int): Instances per mini-batch.
            learning_rate (float): Adam learning rate.
            embed_l2, embed_l1 (float): Regularization on the word table and on the
                transformed entity and context tables.
            layer_l2, layer_l1 (float): Regularization on the weights and biases of
                the convolutions and of the layers after them; the batch-norm and
                entity/context transform parameters are left out.
            max_grad_norm (float): Per-parameter gradient-norm clipping value.
                ``None`` disables clipping.
            metrics, pairwise_metrics (list[str]): Passed to :meth:`run_eval`.
            show_step (int): Print the training loss every ``show_step`` steps.
            model_dir (str): Directory for the ``epoch_<n>`` checkpoints. ``None``
                saves no checkpoint.
            save_epoch (int): Save a checkpoint every ``save_epoch`` epochs.

        Returns:
            object: An instance of self.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for step, (np_batch, _) in enumerate(
                self.iterator.load_data_from_file(train_file, batch_size), 1
            ):
                batch = self._to_tensors(np_batch)
                optimizer.zero_grad(set_to_none=True)
                data_loss = self._data_loss(batch)
                step_loss = data_loss + self._regular_loss(
                    embed_l2, embed_l1, layer_l2, layer_l1
                )
                step_loss.backward()
                if max_grad_norm is not None:
                    for param in self.parameters():
                        if param.grad is not None:
                            nn.utils.clip_grad_norm_(param, max_grad_norm)
                optimizer.step()

                epoch_loss += step_loss.item()
                if step % show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss.item(), data_loss.item()
                        )
                    )

            if model_dir and epoch % save_epoch == 0:
                os.makedirs(model_dir, exist_ok=True)
                torch.save(
                    self.state_dict(), os.path.join(model_dir, "epoch_" + str(epoch))
                )

            eval_info = self._format_metrics(
                self.run_eval(valid_file, batch_size, metrics, pairwise_metrics)
            )
            message = "at epoch {0:d}\ntrain info: loss:{1}\neval info: {2}".format(
                epoch, epoch_loss / step, eval_info
            )
            if test_file is not None:
                message += "\ntest info: " + self._format_metrics(
                    self.run_eval(test_file, batch_size, metrics, pairwise_metrics)
                )
            print(message)

        return self

    @staticmethod
    def _format_metrics(res: dict) -> str:
        return ", ".join(
            "{0}:{1}".format(name, value) for name, value in sorted(res.items())
        )

    @torch.no_grad()
    def run_get_embedding(
        self, infile_name: str, outfile_name: str, batch_size: int = 100
    ):
        """Write the embedding of every article in ``infile_name``.

        Args:
            infile_name (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line per
                article, the format of the news feature file.
            outfile_name (str): Output file, one ``<news_id> <v1,v2,...>`` line per
                article.
            batch_size (int): Articles per mini-batch.

        Returns:
            object: An instance of self.
        """
        self.eval()
        with open(outfile_name, "w") as wt:
            for np_batch, news_ids in self.iterator.load_infer_data_from_file(
                infile_name, batch_size
            ):
                batch = self._to_tensors(np_batch)
                embedding = self._embed_news(batch["words"], batch["entities"])
                for news_id, row in zip(news_ids, embedding.cpu().numpy()):
                    wt.write(news_id + " " + ",".join(map(str, row)) + "\n")
        return self

    def load_model(self, model_path: str):
        """Restore parameters from a ``state_dict`` checkpoint.

        Args:
            model_path (str): Path to the checkpoint file.

        Returns:
            object: An instance of self.
        """
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(state)
        return self


def _fcn_net(
    input_dim: int,
    layer_sizes: list[int],
    activation: nn.Module,
    enable_BN: bool,
    init_method: str,
    init_value: float,
) -> FcnNet:
    """FcnNet whose weights and biases both follow ``init_method``."""

    def init(tensor: torch.Tensor) -> None:
        init_weight_(tensor, init_method, init_value)

    net = FcnNet(
        input_dim, layer_sizes, activation, [0.0] * len(layer_sizes), enable_BN, init
    )
    for linear in (*net.linears, net.out):
        init(linear.bias)
    return net


def _group_by_key(labels: list, preds: list, keys: list) -> tuple[list, list]:
    """Split labels and predictions into one list per key, in first-seen order."""
    groups = {}
    for label, pred, key in zip(labels, preds, keys):
        group_labels, group_preds = groups.setdefault(key, ([], []))
        group_labels.append(label)
        group_preds.append(pred)
    return (
        [group_labels for group_labels, _ in groups.values()],
        [group_preds for _, group_preds in groups.values()],
    )


class DKN(DKNBase):
    """DKN model.

    :Citation:

        H. Wang, F. Zhang, X. Xie and M. Guo, "DKN: Deep Knowledge-Aware Network for
        News Recommendation", in Proceedings of the 2018 World Wide Web Conference on
        World Wide Web, 2018.
    """

    def __init__(
        self,
        news_feature_file: str,
        user_history_file: str,
        word_embedding_file: str,
        entity_embedding_file: str | None = None,
        context_embedding_file: str | None = None,
        history_size: int = 50,
        filter_sizes: list[int] | None = None,
        num_filters: int = 100,
        attention_layer_size: int = 100,
        layer_sizes: list[int] | None = None,
        enable_BN: bool = True,
        init_method: str = "uniform",
        init_value: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Build the DKN model.

        Architecture arguments live on the constructor; training-time knobs (epochs,
        learning rate, batch size, regularization, ...) belong on :meth:`fit`.

        Args:
            news_feature_file (str): One ``<news_id> <w1,w2,...> <e1,e2,...>`` line
                per article.
            user_history_file (str): One ``<user_id> <n1,n2,...>`` line per user.
            word_embedding_file (str): ``.npy`` file of pre-trained
                ``[word_count, dim]`` word embeddings.
            entity_embedding_file (str): ``.npy`` file of pre-trained
                ``[entity_count, entity_dim]`` entity embeddings. ``None`` leaves
                entities out.
            context_embedding_file (str): ``.npy`` file of entity context
                embeddings. ``None`` leaves contexts out; requires
                ``entity_embedding_file``.
            history_size (int): Clicked articles kept per user.
            filter_sizes (list[int]): KCNN window sizes. Defaults to ``[1, 2, 3]``.
            num_filters (int): KCNN filters per window size.
            attention_layer_size (int): Hidden size of the attention network.
            layer_sizes (list[int]): Hidden layer sizes of the scorer. Defaults to
                ``[300]``.
            enable_BN (bool): Whether to use batch normalization in the attention
                network and the scorer.
            init_method (str): ``uniform`` or ``xavier_normal``, the init of the
                attention and scorer weights and biases and of the convolution
                biases. The convolution weights always use a Glorot normal init.
            init_value (float): Range of the ``uniform`` init.
            seed (int): Random seed.
        """
        super().__init__(
            word_embedding_file,
            entity_embedding_file,
            context_embedding_file,
            filter_sizes,
            num_filters,
            init_method,
            init_value,
            seed,
        )
        self.iterator = DKNDataset(news_feature_file, user_history_file, history_size)

        news_dim = self.kcnn.output_dim
        self.attention = _fcn_net(
            2 * news_dim,
            [attention_layer_size],
            nn.ReLU(),
            enable_BN,
            init_method,
            init_value,
        )
        self.scorer = _fcn_net(
            2 * news_dim,
            layer_sizes if layer_sizes is not None else [300],
            nn.Sigmoid(),
            enable_BN,
            init_method,
            init_value,
        )
        self.layer_params += [
            param
            for net in (self.attention, self.scorer)
            for module in (*net.linears, net.out)
            for param in module.parameters()
        ]

        self.to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch: dict) -> torch.Tensor:
        """Click logit ``[B, 1]`` of every (user, candidate article) pair."""
        batch_size, history_size, doc_size = batch["clicked_words"].shape
        # One KCNN pass embeds the candidates and all the clicked articles.
        news = self.kcnn(
            torch.cat(
                [
                    batch["candidate_words"],
                    batch["clicked_words"].reshape(-1, doc_size),
                ]
            ),
            torch.cat(
                [
                    batch["candidate_entities"],
                    batch["clicked_entities"].reshape(-1, doc_size),
                ]
            ),
        )
        candidate = news[:batch_size]
        clicked = news[batch_size:].view(batch_size, history_size, -1)
        user = self._attend(clicked, candidate)
        return self.scorer(torch.cat([user, candidate], dim=1))

    def _attend(self, clicked: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Pool clicked ``[B, H, D]`` into a user vector ``[B, D]``.

        Every history slot takes part in the softmax, the padded ones included.
        """
        pairs = torch.cat([clicked, candidate.unsqueeze(1).expand_as(clicked)], dim=-1)
        logits = self.attention(pairs.flatten(0, 1)).view(*clicked.shape[:2], 1)
        return (clicked * torch.softmax(logits, dim=1)).sum(dim=1)

    def _data_loss(self, batch: dict) -> torch.Tensor:
        """Log loss, with an epsilon that keeps ``log`` finite at 0 and 1."""
        pred = torch.sigmoid(self.forward(batch)).view(-1)
        labels = batch["labels"].view(-1)
        epsilon = 1e-7
        return torch.mean(
            -labels * torch.log(pred + epsilon)
            - (1 - labels) * torch.log(1 - pred + epsilon)
        )

    def _embed_news(self, words: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        return self.kcnn(words, entities)

    @torch.no_grad()
    def run_eval(
        self,
        filename: str,
        batch_size: int = 100,
        metrics: list[str] | None = None,
        pairwise_metrics: list[str] | None = None,
    ) -> dict:
        """Evaluate ``filename`` and return the metric dictionary.

        Args:
            filename (str): A file name that will be evaluated.
            batch_size (int): Instances per mini-batch.
            metrics (list[str]): Metrics over all the instances. Defaults to
                ``["auc"]``.
            pairwise_metrics (list[str]): Metrics averaged over the impressions.
                Defaults to ``["group_auc", "mean_mrr", "ndcg@5;10"]``.

        Returns:
            dict: A dictionary that contains evaluation metrics.
        """
        metrics = metrics if metrics is not None else ["auc"]
        pairwise_metrics = (
            pairwise_metrics
            if pairwise_metrics is not None
            else ["group_auc", "mean_mrr", "ndcg@5;10"]
        )
        self.eval()
        preds, labels, impression_ids = [], [], []
        for np_batch, batch_ids in self.iterator.load_data_from_file(
            filename, batch_size
        ):
            pred = torch.sigmoid(self.forward(self._to_tensors(np_batch)))
            preds.extend(pred.cpu().numpy().reshape(-1))
            labels.extend(np_batch["labels"].reshape(-1))
            impression_ids.extend(batch_ids)

        res = cal_metric(labels, preds, metrics)
        if pairwise_metrics:
            group_labels, group_preds = _group_by_key(labels, preds, impression_ids)
            res.update(cal_metric(group_labels, group_preds, pairwise_metrics))
        return res

    @torch.no_grad()
    def predict(self, infile_name: str, outfile_name: str, batch_size: int = 100):
        """Write the click probability of every instance, one per line.

        Args:
            infile_name (str): Input file name, format is same as train/val/test file.
            outfile_name (str): Output file name, each line is the predict score.
            batch_size (int): Instances per mini-batch.

        Returns:
            object: An instance of self.
        """
        self.eval()
        with open(outfile_name, "w") as wt:
            for np_batch, _ in self.iterator.load_data_from_file(
                infile_name, batch_size
            ):
                pred = torch.sigmoid(self.forward(self._to_tensors(np_batch)))
                wt.write("\n".join(map(str, pred.cpu().numpy().reshape(-1))))
                wt.write("\n")
        return self
