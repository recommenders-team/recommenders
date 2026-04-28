# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)
MODEL_CHECKPOINT = "model.pt"


class NCF(nn.Module):
    """Neural Collaborative Filtering (NCF) implementation

    :Citation:

        He, Xiangnan, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. "Neural collaborative filtering."
        In Proceedings of the 26th International Conference on World Wide Web, pp. 173-182. International World Wide Web
        Conferences Steering Committee, 2017. Link: https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
    """

    def __init__(
        self,
        n_users,
        n_items,
        model_type="NeuMF",
        n_factors=8,
        layer_sizes=[16, 8, 4],
        n_epochs=50,
        batch_size=64,
        learning_rate=5e-3,
        verbose=1,
        seed=None,
    ):
        """Constructor

        Args:
            n_users (int): Number of users in the dataset.
            n_items (int): Number of items in the dataset.
            model_type (str): Model type.
            n_factors (int): Dimension of latent space.
            layer_sizes (list): Number of layers for MLP.
            n_epochs (int): Number of epochs for training.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            verbose (int): Whether to show the training output or not.
            seed (int): Seed.

        """

        super().__init__()

        # seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.seed = seed

        self.n_users = n_users
        self.n_items = n_items
        self.model_type = model_type.lower()
        self.n_factors = n_factors
        self.layer_sizes = layer_sizes
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # check model type
        model_options = ["gmf", "mlp", "neumf"]
        if self.model_type not in model_options:
            raise ValueError(
                "Wrong model type, please select one of this list: {}".format(
                    model_options
                )
            )

        # ncf layer input size
        self.ncf_layer_size = n_factors + layer_sizes[-1]

        # --- Embeddings ---
        # GMF embeddings
        self.embedding_gmf_P = nn.Embedding(n_users, n_factors)
        self.embedding_gmf_Q = nn.Embedding(n_items, n_factors)
        # MLP embeddings
        self.embedding_mlp_P = nn.Embedding(n_users, int(layer_sizes[0] / 2))
        self.embedding_mlp_Q = nn.Embedding(n_items, int(layer_sizes[0] / 2))

        # Initialize embeddings with truncated normal (matches TF truncated_normal stddev=0.01)
        for emb in [
            self.embedding_gmf_P,
            self.embedding_gmf_Q,
            self.embedding_mlp_P,
            self.embedding_mlp_Q,
        ]:
            nn.init.trunc_normal_(emb.weight, mean=0.0, std=0.01)

        # --- MLP layers ---
        mlp_layers = []
        for i in range(1, len(layer_sizes)):
            mlp_layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            mlp_layers.append(nn.ReLU())
        self.mlp_layers = nn.Sequential(*mlp_layers)

        # Initialize MLP weights with Xavier uniform (matches TF VarianceScaling fan_avg uniform)
        # and biases with zeros (matches slim default)
        for module in self.mlp_layers:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        # --- Output layer (no bias, matches TF biases_initializer=None) ---
        if self.model_type == "gmf":
            self.output_layer = nn.Linear(n_factors, 1, bias=False)
        elif self.model_type == "mlp":
            self.output_layer = nn.Linear(layer_sizes[-1], 1, bias=False)
        elif self.model_type == "neumf":
            self.output_layer = nn.Linear(
                n_factors + layer_sizes[-1], 1, bias=False
            )
        nn.init.xavier_uniform_(self.output_layer.weight)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, user_input, item_input):
        """Forward pass

        Args:
            user_input (torch.LongTensor): User indices, shape (batch,).
            item_input (torch.LongTensor): Item indices, shape (batch,).

        Returns:
            torch.Tensor: Predicted scores, shape (batch, 1).
        """

        # GMF path
        gmf_vector = None
        if self.model_type in ("gmf", "neumf"):
            gmf_p = self.embedding_gmf_P(user_input)
            gmf_q = self.embedding_gmf_Q(item_input)
            gmf_vector = gmf_p * gmf_q

        # MLP path
        mlp_vector = None
        if self.model_type in ("mlp", "neumf"):
            mlp_p = self.embedding_mlp_P(user_input)
            mlp_q = self.embedding_mlp_Q(item_input)
            mlp_vector = self.mlp_layers(torch.cat([mlp_p, mlp_q], dim=-1))

        # Output
        if self.model_type == "gmf":
            output = self.output_layer(gmf_vector)
        elif self.model_type == "mlp":
            output = self.output_layer(mlp_vector)
        else:  # neumf
            output = self.output_layer(
                torch.cat([gmf_vector, mlp_vector], dim=-1)
            )

        return torch.sigmoid(output)

    def save(self, dir_name):
        """Save model parameters in `dir_name`

        Args:
            dir_name (str): directory name, which should be a folder name instead of file name
                we will create a new directory if not existing.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.state_dict(), os.path.join(dir_name, MODEL_CHECKPOINT))

    def load(self, gmf_dir=None, mlp_dir=None, neumf_dir=None, alpha=0.5):
        """Load model parameters for further use.

        GMF model --> load parameters in `gmf_dir`

        MLP model --> load parameters in `mlp_dir`

        NeuMF model --> load parameters in `neumf_dir` or in `gmf_dir` and `mlp_dir`

        Args:
            gmf_dir (str): Directory name for GMF model.
            mlp_dir (str): Directory name for MLP model.
            neumf_dir (str): Directory name for neumf model.
            alpha (float): the concatenation hyper-parameter for gmf and mlp output layer.

        Returns:
            object: Load parameters in this model.
        """

        if self.model_type == "gmf" and gmf_dir is not None:
            state_dict = torch.load(
                os.path.join(gmf_dir, MODEL_CHECKPOINT),
                map_location=self.device,
            )
            self.load_state_dict(state_dict)

        elif self.model_type == "mlp" and mlp_dir is not None:
            state_dict = torch.load(
                os.path.join(mlp_dir, MODEL_CHECKPOINT),
                map_location=self.device,
            )
            self.load_state_dict(state_dict)

        elif self.model_type == "neumf" and neumf_dir is not None:
            state_dict = torch.load(
                os.path.join(neumf_dir, MODEL_CHECKPOINT),
                map_location=self.device,
            )
            self.load_state_dict(state_dict)

        elif self.model_type == "neumf" and gmf_dir is not None and mlp_dir is not None:
            self._load_neumf(gmf_dir, mlp_dir, alpha)

        else:
            raise NotImplementedError

    def _load_neumf(self, gmf_dir, mlp_dir, alpha):
        """Load gmf and mlp model parameters for further use in NeuMF.
        NeuMF model --> load parameters in `gmf_dir` and `mlp_dir`
        """
        gmf_state = torch.load(
            os.path.join(gmf_dir, MODEL_CHECKPOINT),
            map_location=self.device,
        )
        mlp_state = torch.load(
            os.path.join(mlp_dir, MODEL_CHECKPOINT),
            map_location=self.device,
        )

        # Build a new state dict to load atomically
        new_state = self.state_dict()

        # GMF embeddings from GMF model
        new_state["embedding_gmf_P.weight"] = gmf_state["embedding_gmf_P.weight"]
        new_state["embedding_gmf_Q.weight"] = gmf_state["embedding_gmf_Q.weight"]

        # MLP embeddings from MLP model
        new_state["embedding_mlp_P.weight"] = mlp_state["embedding_mlp_P.weight"]
        new_state["embedding_mlp_Q.weight"] = mlp_state["embedding_mlp_Q.weight"]

        # MLP layer weights from MLP model
        for key in mlp_state:
            if key.startswith("mlp_layers."):
                new_state[key] = mlp_state[key]

        # Concatenate output layer weights: [alpha * gmf_fc, (1-alpha) * mlp_fc]
        # PyTorch Linear weight shape is (out_features, in_features), so concat on dim=1
        new_state["output_layer.weight"] = torch.cat(
            [
                alpha * gmf_state["output_layer.weight"],
                (1 - alpha) * mlp_state["output_layer.weight"],
            ],
            dim=1,
        )

        self.load_state_dict(new_state)

    def fit(self, data):
        """Fit model with training data

        Args:
            data (NCFDataset): initilized Dataset in ./dataset.py
        """

        # get user and item mapping dict
        self.user2id = data.user2id
        self.item2id = data.item2id
        self.id2user = data.id2user
        self.id2item = data.id2item

        # create optimizer AFTER model is on device
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        # loop for n_epochs
        for epoch_count in range(1, self.n_epochs + 1):

            # negative sampling for training
            train_begin = time()

            # initialize
            train_loss = []
            self.train()

            # calculate loss and update NCF parameters
            for user_input, item_input, labels in data.train_loader(self.batch_size):

                user_input = np.array([self.user2id[x] for x in user_input])
                item_input = np.array([self.item2id[x] for x in item_input])
                labels = np.array(labels)

                user_tensor = torch.LongTensor(user_input).to(self.device)
                item_tensor = torch.LongTensor(item_input).to(self.device)
                label_tensor = (
                    torch.FloatTensor(labels).unsqueeze(1).to(self.device)
                )

                # forward pass
                output = self.forward(user_tensor, item_tensor)
                loss = criterion(output, label_tensor)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
            train_time = time() - train_begin

            # output every self.verbose
            if self.verbose and epoch_count % self.verbose == 0:
                logger.info(
                    "Epoch %d [%.2fs]: train_loss = %.6f "
                    % (epoch_count, train_time, sum(train_loss) / len(train_loss))
                )

    def predict(self, user_input, item_input, is_list=False):
        """Predict function of this trained model

        Args:
            user_input (list or element of list): userID or userID list
            item_input (list or element of list): itemID or itemID list
            is_list (bool): if true, the input is list type
                noting that list-wise type prediction is faster than element-wise's.

        Returns:
            list or float: A list of predicted rating or predicted rating score.
        """

        if is_list:
            output = self._predict(user_input, item_input)
            return list(output.reshape(-1))

        else:
            output = self._predict(np.array([user_input]), np.array([item_input]))
            return float(output.reshape(-1)[0])

    def _predict(self, user_input, item_input):

        # index converting
        user_input = np.array([self.user2id[x] for x in user_input])
        item_input = np.array([self.item2id[x] for x in item_input])

        user_tensor = torch.LongTensor(user_input).to(self.device)
        item_tensor = torch.LongTensor(item_input).to(self.device)

        self.eval()
        with torch.no_grad():
            output = self.forward(user_tensor, item_tensor)

        return output.cpu().numpy()
