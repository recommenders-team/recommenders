# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn as nn

from recommenders.models.sasrec.model import SASREC, Encoder, LayerNormalization
from recommenders.models.sasrec.model import pad_sequences


class SSEPT(SASREC):
    """
    SSE-PT Model

    :Citation:

    Wu L., Li S., Hsieh C-J., Sharpnack J., SSE-PT: Sequential Recommendation
    Via Personalized Transformer, RecSys, 2020.
    TF 1.x codebase: https://github.com/SSE-PT/SSE-PT
    TF 2.x codebase (SASREc): https://github.com/nnkkmto/SASRec-tf2
    """

    def __init__(self, **kwargs):
        """Model initialization.

        Args:
            item_num (int): Number of items in the dataset.
            seq_max_len (int): Maximum number of items in user history.
            num_blocks (int): Number of Transformer blocks to be used.
            embedding_dim (int): Item embedding dimension.
            attention_dim (int): Transformer attention dimension.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout rate.
            l2_reg (float): Coefficient of the L2 regularization.
            num_neg_test (int): Number of negative examples used in testing.
            user_num (int): Number of users in the dataset.
            user_embedding_dim (int): User embedding dimension.
            item_embedding_dim (int): Item embedding dimension.
        """
        super().__init__(**kwargs)

        self.user_num = kwargs.get("user_num", None)  # New
        self.conv_dims = kwargs.get("conv_dims", [200, 200])  # modified
        self.user_embedding_dim = kwargs.get(
            "user_embedding_dim", self.embedding_dim
        )  # extra
        self.item_embedding_dim = kwargs.get("item_embedding_dim", self.embedding_dim)
        self.hidden_units = self.item_embedding_dim + self.user_embedding_dim

        # New, user embedding
        self.user_embedding_layer = nn.Embedding(
            self.user_num + 1,
            self.user_embedding_dim,
            padding_idx=0,
        )

        self.positional_embedding_layer = nn.Embedding(
            self.seq_max_len,
            self.user_embedding_dim + self.item_embedding_dim,  # difference
        )

        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.hidden_units,
            self.hidden_units,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.hidden_units, 1e-08
        )

        # Re-initialize weights for the new layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to match TensorFlow/Keras defaults.

        Overrides parent to also initialize user_embedding_layer.
        """
        # Call parent initialization
        super()._init_weights()

        # Initialize user embedding with uniform distribution matching TF default
        nn.init.uniform_(self.user_embedding_layer.weight, -0.05, 0.05)
        # Keep padding_idx as zeros
        with torch.no_grad():
            self.user_embedding_layer.weight[0].fill_(0)

        # Re-initialize positional embeddings (which has different dim in SSEPT)
        nn.init.uniform_(self.positional_embedding_layer.weight, -0.05, 0.05)

    def forward(self, x, training=True):
        """Model forward pass.

        Args:
            x (dict): Input dictionary containing 'users', 'input_seq', 'positive', 'negative'.
            training (bool): Training mode flag.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor:
            - Logits of the positive examples.
            - Logits of the negative examples.
            - Mask for nonzero targets
        """
        users = x["users"]
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = (input_seq != 0).float().unsqueeze(-1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # User Encoding
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)

        # replicate the user embedding for all the items
        u_latent = u_latent.expand(-1, input_seq.size(1), -1)  # (b, s, h)

        seq_embeddings = torch.cat([seq_embeddings, u_latent], dim=2).reshape(
            input_seq.size(0), -1, self.hidden_units
        )
        seq_embeddings = seq_embeddings + positional_embeddings

        # dropout
        if training:
            seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings = seq_embeddings * mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, s, h1 + h2)

        seq_attention = self.encoder(seq_attention, training=training, mask=mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = pos * (pos != 0).long()  # masking
        neg = neg * (neg != 0).long()  # masking

        user_emb = u_latent.reshape(
            input_seq.size(0) * self.seq_max_len, self.user_embedding_dim
        )
        pos = pos.reshape(input_seq.size(0) * self.seq_max_len)
        neg = neg.reshape(input_seq.size(0) * self.seq_max_len)
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)

        # Add user embeddings
        pos_emb = torch.cat([pos_emb, user_emb], dim=1).reshape(-1, self.hidden_units)
        neg_emb = torch.cat([neg_emb, user_emb], dim=1).reshape(-1, self.hidden_units)

        seq_emb = seq_attention.reshape(
            input_seq.size(0) * self.seq_max_len, self.hidden_units
        )  # (b*s, d)

        pos_logits = (pos_emb * seq_emb).sum(dim=-1)
        neg_logits = (neg_emb * seq_emb).sum(dim=-1)

        pos_logits = pos_logits.unsqueeze(-1)  # (bs, 1)
        neg_logits = neg_logits.unsqueeze(-1)  # (bs, 1)

        # masking for loss calculation
        istarget = (pos != 0).float()

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        """
        Model prediction for candidate (negative) items.

        Args:
            inputs (dict): Input dictionary containing 'user', 'input_seq', 'candidate'.
                - user: (batch, 1) tensor of user indices
                - input_seq: (batch, seq_max_len) tensor of item indices
                - candidate: (batch, num_candidates) tensor of candidate item indices

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_candidates).
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            user = inputs["user"]
            input_seq = inputs["input_seq"]
            candidate = inputs["candidate"]

            # Convert numpy arrays to tensors if necessary
            if not isinstance(user, torch.Tensor):
                user = torch.LongTensor(user).to(device)
            if not isinstance(input_seq, torch.Tensor):
                input_seq = torch.LongTensor(input_seq).to(device)
            if not isinstance(candidate, torch.Tensor):
                candidate = torch.LongTensor(candidate).to(device)

            num_candidates = candidate.size(1)

            mask = (input_seq != 0).float().unsqueeze(-1)
            seq_embeddings, positional_embeddings = self.embedding(input_seq)  # (b, s, item_dim)

            # User embedding for sequence
            u_latent = self.user_embedding_layer(user)  # (b, 1, user_dim)
            u_latent = u_latent * (self.user_embedding_dim ** 0.5)
            u_latent = u_latent.expand(-1, input_seq.size(1), -1)  # (b, s, user_dim)

            # Concatenate item and user embeddings
            seq_embeddings = torch.cat([seq_embeddings, u_latent], dim=2)  # (b, s, hidden_units)
            seq_embeddings = seq_embeddings + positional_embeddings  # (b, s, hidden_units)

            seq_embeddings = seq_embeddings * mask
            seq_attention = self.encoder(seq_embeddings, training=False, mask=mask)
            seq_attention = self.layer_normalization(seq_attention)  # (b, s, hidden_units)

            # Take only the last position embedding for each sequence
            seq_emb = seq_attention[:, -1, :]  # (b, hidden_units)

            # Get candidate item embeddings
            candidate_item_emb = self.item_embedding_layer(candidate)  # (b, num_cand, item_dim)

            # User embedding for candidates (same user for all candidates in each batch)
            user_emb_for_cand = self.user_embedding_layer(user)  # (b, 1, user_dim)
            user_emb_for_cand = user_emb_for_cand * (self.user_embedding_dim ** 0.5)
            user_emb_for_cand = user_emb_for_cand.expand(-1, num_candidates, -1)  # (b, num_cand, user_dim)

            # Concatenate item and user embeddings for candidates
            candidate_emb = torch.cat([candidate_item_emb, user_emb_for_cand], dim=2)  # (b, num_cand, hidden_units)

            # Compute logits via batched dot product
            # (b, num_cand, hidden_units) * (b, 1, hidden_units) -> (b, num_cand, hidden_units) -> sum -> (b, num_cand)
            test_logits = (candidate_emb * seq_emb.unsqueeze(1)).sum(dim=-1)

        return test_logits


    def create_combined_dataset(self, u, seq, pos, neg):
        """
        function to create model inputs from sampled batch data.
        This function is used only during training.
        Overrides parent to include users in the inputs.
        """
        inputs = {}
        seq = pad_sequences(seq, padding="pre", truncating="pre", maxlen=self.seq_max_len)
        pos = pad_sequences(pos, padding="pre", truncating="pre", maxlen=self.seq_max_len)
        neg = pad_sequences(neg, padding="pre", truncating="pre", maxlen=self.seq_max_len)

        inputs["users"] = np.expand_dims(np.array(u), axis=-1)
        inputs["input_seq"] = seq
        inputs["positive"] = pos
        inputs["negative"] = neg

        target = np.concatenate(
            [
                np.repeat(1, seq.shape[0] * seq.shape[1]),
                np.repeat(0, seq.shape[0] * seq.shape[1]),
            ],
            axis=0,
        )
        target = np.expand_dims(target, axis=-1)
        return inputs, target

    # train_model is inherited from SASREC (handles "users" key dynamically)
