# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from recommenders.utils.timer import Timer


class MultiHeadAttention(nn.Module):
    """
    - Q (query), K (key) and V (value) are split into multiple heads (num_heads)
    - each tuple (q, k, v) are fed to scaled_dot_product_attention
    - all attention outputs are concatenated
    """

    def __init__(self, attention_dim, num_heads, dropout_rate):
        """Initialize parameters.

        Args:
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            dropout_rate (float): Dropout probability.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        assert attention_dim % self.num_heads == 0
        self.dropout_rate = dropout_rate

        self.depth = attention_dim // self.num_heads

        self.Q = nn.Linear(self.attention_dim, self.attention_dim, bias=True)
        self.K = nn.Linear(self.attention_dim, self.attention_dim, bias=True)
        self.V = nn.Linear(self.attention_dim, self.attention_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, queries, keys):
        """Model forward pass.

        Args:
            queries (torch.Tensor): Tensor of queries.
            keys (torch.Tensor): Tensor of keys

        Returns:
            torch.Tensor: Output tensor.
        """
        # Linear projections
        Q = self.Q(queries)  # (N, T_q, C)
        K = self.K(keys)  # (N, T_k, C)
        V = self.V(keys)  # (N, T_k, C)

        # --- MULTI HEAD ---
        # Split and concat, Q_, K_ and V_ are all (h*N, T_q, C/h)
        Q_ = torch.cat(torch.split(Q, self.depth, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.depth, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.depth, dim=2), dim=0)

        # --- SCALED DOT PRODUCT ---
        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (self.depth ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(keys.sum(dim=-1)))  # (N, T_k)
        key_masks = key_masks.repeat(self.num_heads, 1)  # (h*N, T_k)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-(2**32) + 1)
        outputs = torch.where(key_masks == 0, paddings, outputs)

        # Future blinding (Causality)
        diag_vals = torch.ones_like(outputs[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        masks = tril.unsqueeze(0).repeat(outputs.size(0), 1, 1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(masks) * (-(2**32) + 1)
        outputs = torch.where(masks == 0, paddings, outputs)

        # Activation
        outputs = F.softmax(outputs, dim=-1)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(queries.sum(dim=-1)))  # (N, T_q)
        query_masks = query_masks.repeat(self.num_heads, 1)  # (h*N, T_q)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))  # (h*N, T_q, T_k)
        outputs = outputs * query_masks

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # (h*N, T_q, C/h)

        # --- MULTI HEAD ---
        # concat heads
        outputs = torch.cat(torch.split(outputs, outputs.size(0) // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs = outputs + queries

        return outputs


class PointWiseFeedForward(nn.Module):
    """
    Convolution layers with residual connection
    """

    def __init__(self, embedding_dim, conv_dims, dropout_rate):
        """Initialize parameters.

        Args:
            embedding_dim (int): Embedding dimension (input channels).
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(PointWiseFeedForward, self).__init__()
        self.conv_dims = conv_dims
        self.dropout_rate = dropout_rate
        # Conv1d in PyTorch expects (batch, channels, seq_len)
        self.conv_layer1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=self.conv_dims[0],
            kernel_size=1,
            bias=True
        )
        self.conv_layer2 = nn.Conv1d(
            in_channels=self.conv_dims[0],
            out_channels=self.conv_dims[1],
            kernel_size=1,
            bias=True
        )
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, channels).

        Returns:
            torch.Tensor: Output tensor.
        """
        # Transpose for Conv1d: (batch, seq_len, channels) -> (batch, channels, seq_len)
        output = x.transpose(1, 2)

        output = self.conv_layer1(output)
        output = F.relu(output)
        output = self.dropout_layer(output)

        output = self.conv_layer2(output)
        output = self.dropout_layer(output)

        # Transpose back: (batch, channels, seq_len) -> (batch, seq_len, channels)
        output = output.transpose(1, 2)

        # Residual connection
        output = output + x

        return output


class EncoderLayer(nn.Module):
    """
    Transformer based encoder layer
    """

    def __init__(
        self,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim

        self.mha = MultiHeadAttention(attention_dim, num_heads, dropout_rate)
        self.ffn = PointWiseFeedForward(embedding_dim, conv_dims, dropout_rate)

        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

    def call_(self, x, training, mask):
        """Model forward pass (alternative implementation).

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): Training mode flag.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        attn_output = self.mha(queries=self.layer_normalization(x), keys=x)
        if training:
            attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)

        # feed forward network
        ffn_output = self.ffn(out1)
        if training:
            ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        # masking
        out2 = out2 * mask

        return out2

    def forward(self, x, training, mask):
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): True if in training mode.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_norm = self.layer_normalization(x)
        attn_output = self.mha(queries=x_norm, keys=x)
        attn_output = self.ffn(attn_output)
        out = attn_output * mask

        return out


class Encoder(nn.Module):
    """
    Invokes Transformer based encoder with user defined number of layers
    """

    def __init__(
        self,
        num_layers,
        seq_max_len,
        embedding_dim,
        attention_dim,
        num_heads,
        conv_dims,
        dropout_rate,
    ):
        """Initialize parameters.

        Args:
            num_layers (int): Number of layers.
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            attention_dim (int): Dimension of the attention embeddings.
            num_heads (int): Number of heads in the multi-head self-attention module.
            conv_dims (list): List of the dimensions of the Feedforward layer.
            dropout_rate (float): Dropout probability.
        """
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                seq_max_len,
                embedding_dim,
                attention_dim,
                num_heads,
                conv_dims,
                dropout_rate,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, training, mask):
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor.
            training (bool): True if in training mode.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x  # (batch_size, input_seq_len, d_model)


class LayerNormalization(nn.Module):
    """
    Layer normalization using mean and variance
    gamma and beta are the learnable parameters
    """

    def __init__(self, seq_max_len, embedding_dim, epsilon):
        """Initialize parameters.

        Args:
            seq_max_len (int): Maximum sequence length.
            embedding_dim (int): Embedding dimension.
            epsilon (float): Epsilon value.
        """
        super(LayerNormalization, self).__init__()
        self.seq_max_len = seq_max_len
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.params_shape = (self.seq_max_len, self.embedding_dim)
        self.gamma = nn.Parameter(torch.ones(self.params_shape))
        self.beta = nn.Parameter(torch.zeros(self.params_shape))

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / ((variance + self.epsilon) ** 0.5)
        output = self.gamma * normalized + self.beta
        return output


def pad_sequences(sequences, maxlen, padding='pre', truncating='pre', value=0):
    """Pads sequences to the same length.

    Replacement for tf.keras.preprocessing.sequence.pad_sequences

    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Maximum length of all sequences.
        padding: 'pre' or 'post' - pad either before or after each sequence.
        truncating: 'pre' or 'post' - remove values from sequences larger than maxlen.
        value: Padding value.

    Returns:
        numpy.ndarray: Padded sequences.
    """
    result = np.full((len(sequences), maxlen), value, dtype=np.int64)

    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
        if truncating == 'pre':
            trunc = seq[-maxlen:]
        else:
            trunc = seq[:maxlen]

        if padding == 'pre':
            result[i, -len(trunc):] = trunc
        else:
            result[i, :len(trunc)] = trunc

    return result


class SASREC(nn.Module):
    """SAS Rec model
    Self-Attentive Sequential Recommendation Using Transformer

    :Citation:

        Wang-Cheng Kang, Julian McAuley (2018), Self-Attentive Sequential
        Recommendation. Proceedings of IEEE International Conference on
        Data Mining (ICDM'18)

        Original source code from nnkkmto/SASRec-tf2,
        https://github.com/nnkkmto/SASRec-tf2

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
        """
        super(SASREC, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.num_blocks = kwargs.get("num_blocks", 2)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.attention_num_heads = kwargs.get("attention_num_heads", 1)
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.num_neg_test = kwargs.get("num_neg_test", 100)

        self.item_embedding_layer = nn.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            padding_idx=0,
        )

        self.positional_embedding_layer = nn.Embedding(
            self.seq_max_len,
            self.embedding_dim,
        )
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.embedding_dim,
            self.attention_dim,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.embedding_dim, 1e-08
        )

        # Initialize weights to match TensorFlow defaults
        # Only call if this is SASREC directly (not a subclass like SSEPT)
        if type(self) is SASREC:
            self._init_weights()

    def _init_weights(self):
        """Initialize weights to match TensorFlow/Keras defaults.

        TensorFlow Embedding uses uniform(-0.05, 0.05) by default.
        TensorFlow Dense uses Glorot uniform initialization.
        """
        # Initialize embeddings with uniform distribution matching TF default
        nn.init.uniform_(self.item_embedding_layer.weight, -0.05, 0.05)
        # Keep padding_idx as zeros
        with torch.no_grad():
            self.item_embedding_layer.weight[0].fill_(0)

        nn.init.uniform_(self.positional_embedding_layer.weight, -0.05, 0.05)

        # Initialize Linear layers (Dense in TF) with Glorot/Xavier uniform
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def embedding(self, input_seq):
        """Compute the sequence and positional embeddings.

        Args:
            input_seq (torch.Tensor): Input sequence

        Returns:
            torch.Tensor, torch.Tensor:
            - Sequence embeddings.
            - Positional embeddings.
        """
        seq_embeddings = self.item_embedding_layer(input_seq)
        seq_embeddings = seq_embeddings * (self.embedding_dim ** 0.5)

        positional_seq = torch.arange(input_seq.size(1), device=input_seq.device).unsqueeze(0)
        positional_seq = positional_seq.expand(input_seq.size(0), -1)
        positional_embeddings = self.positional_embedding_layer(positional_seq)

        return seq_embeddings, positional_embeddings

    def forward(self, x, training=True):
        """Model forward pass.

        Args:
            x (dict): Input dictionary containing 'input_seq', 'positive', 'negative'.
            training (bool): Training mode flag.

        Returns:
            torch.Tensor, torch.Tensor, torch.Tensor:
            - Logits of the positive examples.
            - Logits of the negative examples.
            - Mask for nonzero targets
        """
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = (input_seq != 0).float().unsqueeze(-1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # add positional embeddings
        seq_embeddings = seq_embeddings + positional_embeddings

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings)

        # masking
        seq_embeddings = seq_embeddings * mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = pos * (pos != 0).long()  # masking
        neg = neg * (neg != 0).long()  # masking

        pos = pos.reshape(input_seq.size(0) * self.seq_max_len)
        neg = neg.reshape(input_seq.size(0) * self.seq_max_len)
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)
        seq_emb = seq_attention.reshape(
            input_seq.size(0) * self.seq_max_len, self.embedding_dim
        )  # (b*s, d)

        pos_logits = (pos_emb * seq_emb).sum(dim=-1)
        neg_logits = (neg_emb * seq_emb).sum(dim=-1)

        pos_logits = pos_logits.unsqueeze(-1)  # (bs, 1)
        neg_logits = neg_logits.unsqueeze(-1)  # (bs, 1)

        # masking for loss calculation
        istarget = (pos != 0).float()

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        """Returns the logits for the test items.

        Args:
            inputs (dict): Input dictionary containing 'input_seq', 'candidate'.
                - input_seq: (batch, seq_max_len) tensor of item indices
                - candidate: (batch, num_candidates) tensor of candidate item indices

        Returns:
            torch.Tensor: Output tensor of shape (batch, num_candidates).
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            input_seq = inputs["input_seq"]
            candidate = inputs["candidate"]

            # Convert numpy arrays to tensors if necessary
            if not isinstance(input_seq, torch.Tensor):
                input_seq = torch.LongTensor(input_seq).to(device)
            if not isinstance(candidate, torch.Tensor):
                candidate = torch.LongTensor(candidate).to(device)

            mask = (input_seq != 0).float().unsqueeze(-1)
            seq_embeddings, positional_embeddings = self.embedding(input_seq)
            seq_embeddings = seq_embeddings + positional_embeddings
            seq_embeddings = seq_embeddings * mask
            seq_attention = seq_embeddings
            seq_attention = self.encoder(seq_attention, training=False, mask=mask)
            seq_attention = self.layer_normalization(seq_attention)  # (b, s, d)

            # Take only the last position embedding for each sequence
            seq_emb = seq_attention[:, -1, :]  # (b, d)

            # Get candidate embeddings
            candidate_emb = self.item_embedding_layer(candidate)  # (b, num_cand, d)

            # Compute logits via batched dot product
            # (b, num_cand, d) * (b, 1, d) -> (b, num_cand, d) -> sum -> (b, num_cand)
            test_logits = (candidate_emb * seq_emb.unsqueeze(1)).sum(dim=-1)

        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        """Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).

        Args:
            pos_logits (torch.Tensor): Logits of the positive examples.
            neg_logits (torch.Tensor): Logits of the negative examples.
            istarget (torch.Tensor): Mask for nonzero targets.

        Returns:
            torch.Tensor: Loss.
        """
        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # ignore padding items (0)
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        # L2 regularization is handled by optimizer weight_decay
        return loss

    def create_combined_dataset(self, u, seq, pos, neg):
        """
        function to create model inputs from sampled batch data.
        This function is used only during training.
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

    def train_model(
        self,
        dataset,
        sampler,
        num_epochs=10,
        batch_size=128,
        learning_rate=0.001,
        val_epoch=0,
        eval_batch_size=256,
        verbose=True,
    ):
        """
        Train the model.

        Args:
            dataset: The dataset object containing user_train, user_valid, user_test.
            sampler: WarpSampler instance for generating training batches.
            num_epochs (int): Number of training epochs. Default: 10.
            batch_size (int): Training batch size. Default: 128.
            learning_rate (float): Learning rate for Adam optimizer. Default: 0.001.
            val_epoch (int): Evaluate on validation set every N epochs.
                Set to 0 to disable validation during training. Default: 0.
            eval_batch_size (int): Batch size for evaluation. Default: 256.
            verbose (bool): Print training progress. Default: True.

        Returns:
            dict: Training history containing:
                - 'loss': List of average loss per epoch
                - 'val_ndcg': List of validation NDCG@10 (if val_epoch > 0)
                - 'val_hr': List of validation HR@10 (if val_epoch > 0)
        """
        num_steps = int(len(dataset.user_train) / batch_size)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=self.l2_reg,
        )

        # Training history
        history = {
            "loss": [],
            "val_ndcg": [],
            "val_hr": [],
        }

        T = 0.0
        t0 = Timer()
        t0.start()

        for epoch in range(1, num_epochs + 1):
            step_loss = []
            self.train()  # Set training mode

            for step in tqdm(
                range(num_steps), total=num_steps, ncols=70, leave=False, unit="b",
                desc=f"Epoch {epoch}/{num_epochs}" if verbose else None,
                disable=not verbose,
            ):
                u, seq, pos, neg = sampler.next_batch()

                inputs, target = self.create_combined_dataset(u, seq, pos, neg)

                # Convert to tensors and move to device
                inp = {}
                for key in ["users", "input_seq", "positive", "negative"]:
                    if key in inputs:
                        inp[key] = torch.LongTensor(inputs[key]).to(device)

                optimizer.zero_grad()
                pos_logits, neg_logits, loss_mask = self(inp, training=True)
                loss = self.loss_function(pos_logits, neg_logits, loss_mask)
                loss.backward()
                optimizer.step()

                step_loss.append(loss.item())

            # Record average epoch loss
            avg_loss = np.mean(step_loss)
            history["loss"].append(avg_loss)

            # Validation evaluation (if enabled)
            if val_epoch > 0 and epoch % val_epoch == 0:
                t0.stop()
                T += t0.interval

                val_metrics = self.evaluate_valid(dataset, eval_batch_size=eval_batch_size)
                history["val_ndcg"].append(val_metrics[0])
                history["val_hr"].append(val_metrics[1])

                if verbose:
                    print(
                        f"Epoch {epoch}: loss={avg_loss:.4f}, "
                        f"val_NDCG@10={val_metrics[0]:.4f}, val_HR@10={val_metrics[1]:.4f}, "
                        f"time={T:.1f}s"
                    )
                t0.start()
            elif verbose:
                print(f"Epoch {epoch}: loss={avg_loss:.4f}")

        t0.stop()
        if verbose:
            print(f"Training complete. Total time: {T + t0.interval:.1f}s")

        return history

    # Alias for backward compatibility
    def train(self, mode=True):
        """Sets the module in training mode.

        This method is overridden to maintain compatibility with both PyTorch's
        nn.Module.train() and the original train() method for training the model.

        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).

        Returns:
            SASREC: Returns self.
        """
        return super().train(mode)

    def evaluate(self, dataset, seed=None, eval_batch_size=256):
        """
        Evaluation on the test users (users with at least 3 items)

        Args:
            dataset: The dataset object containing user_train, user_valid, user_test.
            seed (int, optional): Random seed for reproducibility. If None, results may vary.
            eval_batch_size (int): Batch size for evaluation. Default: 256.

        Returns:
            tuple: (NDCG@10, Hit@10) metrics.
        """
        self.eval()
        device = next(self.parameters()).device

        usernum = dataset.usernum
        itemnum = dataset.itemnum
        train = dataset.user_train
        valid = dataset.user_valid
        test = dataset.user_test

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = list(range(1, usernum + 1))

        # Filter valid users (those with train and test data)
        valid_users = [u for u in users if len(train[u]) >= 1 and len(test[u]) >= 1]

        NDCG = 0.0
        HT = 0.0

        # Process in batches
        for batch_start in tqdm(range(0, len(valid_users), eval_batch_size),
                                ncols=70, leave=False, unit="batch"):
            batch_users = valid_users[batch_start:batch_start + eval_batch_size]
            batch_size = len(batch_users)

            # Pre-allocate arrays for batch
            seqs = np.zeros((batch_size, self.seq_max_len), dtype=np.int64)
            candidates = np.zeros((batch_size, 1 + self.num_neg_test), dtype=np.int64)

            for i, u in enumerate(batch_users):
                # Build sequence: train items + last valid item
                idx = self.seq_max_len - 1
                if len(valid[u]) > 0:
                    seqs[i, idx] = valid[u][0]
                    idx -= 1
                for item in reversed(train[u]):
                    if idx < 0:
                        break
                    seqs[i, idx] = item
                    idx -= 1

                # Build candidates: test item + negative samples
                rated = set(train[u])
                rated.add(0)
                candidates[i, 0] = test[u][0]

                # Vectorized negative sampling
                neg_samples = []
                while len(neg_samples) < self.num_neg_test:
                    samples = np.random.randint(1, itemnum + 1, size=self.num_neg_test * 2)
                    for s in samples:
                        if s not in rated:
                            neg_samples.append(s)
                            if len(neg_samples) >= self.num_neg_test:
                                break
                candidates[i, 1:] = neg_samples[:self.num_neg_test]

            # Convert to tensors and run batch prediction
            # Include user IDs for SSEPT compatibility (ignored by SASREC.predict)
            user_ids = np.array(batch_users, dtype=np.int64).reshape(-1, 1)
            inputs = {
                "user": torch.LongTensor(user_ids).to(device),
                "input_seq": torch.LongTensor(seqs).to(device),
                "candidate": torch.LongTensor(candidates).to(device),
            }

            with torch.no_grad():
                predictions = -1.0 * self.predict(inputs)
                predictions = predictions.cpu().numpy()

            # Compute metrics for batch
            for i in range(batch_size):
                rank = predictions[i].argsort().argsort()[0]
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1

        valid_user = len(valid_users)
        return NDCG / valid_user, HT / valid_user

    def evaluate_valid(self, dataset, seed=None, eval_batch_size=256):
        """
        Evaluation on the validation users

        Args:
            dataset: The dataset object containing user_train, user_valid.
            seed (int, optional): Random seed for reproducibility. If None, results may vary.
            eval_batch_size (int): Batch size for evaluation. Default: 256.

        Returns:
            tuple: (NDCG@10, Hit@10) metrics.
        """
        self.eval()
        device = next(self.parameters()).device

        usernum = dataset.usernum
        itemnum = dataset.itemnum
        train = dataset.user_train
        valid = dataset.user_valid

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if usernum > 10000:
            users = random.sample(range(1, usernum + 1), 10000)
        else:
            users = list(range(1, usernum + 1))

        # Filter valid users (those with train and valid data)
        valid_users = [u for u in users if len(train[u]) >= 1 and len(valid[u]) >= 1]

        NDCG = 0.0
        HT = 0.0

        # Process in batches
        for batch_start in tqdm(range(0, len(valid_users), eval_batch_size),
                                ncols=70, leave=False, unit="batch"):
            batch_users = valid_users[batch_start:batch_start + eval_batch_size]
            batch_size = len(batch_users)

            # Pre-allocate arrays for batch
            seqs = np.zeros((batch_size, self.seq_max_len), dtype=np.int64)
            candidates = np.zeros((batch_size, 1 + self.num_neg_test), dtype=np.int64)

            for i, u in enumerate(batch_users):
                # Build sequence: only train items (no valid item for validation eval)
                idx = self.seq_max_len - 1
                for item in reversed(train[u]):
                    if idx < 0:
                        break
                    seqs[i, idx] = item
                    idx -= 1

                # Build candidates: valid item + negative samples
                rated = set(train[u])
                rated.add(0)
                candidates[i, 0] = valid[u][0]

                # Vectorized negative sampling
                neg_samples = []
                while len(neg_samples) < self.num_neg_test:
                    samples = np.random.randint(1, itemnum + 1, size=self.num_neg_test * 2)
                    for s in samples:
                        if s not in rated:
                            neg_samples.append(s)
                            if len(neg_samples) >= self.num_neg_test:
                                break
                candidates[i, 1:] = neg_samples[:self.num_neg_test]

            # Convert to tensors and run batch prediction
            # Include user IDs for SSEPT compatibility (ignored by SASREC.predict)
            user_ids = np.array(batch_users, dtype=np.int64).reshape(-1, 1)
            inputs = {
                "user": torch.LongTensor(user_ids).to(device),
                "input_seq": torch.LongTensor(seqs).to(device),
                "candidate": torch.LongTensor(candidates).to(device),
            }

            with torch.no_grad():
                predictions = -1.0 * self.predict(inputs)
                predictions = predictions.cpu().numpy()

            # Compute metrics for batch
            for i in range(batch_size):
                rank = predictions[i].argsort().argsort()[0]
                if rank < 10:
                    NDCG += 1 / np.log2(rank + 2)
                    HT += 1

        valid_user = len(valid_users)
        return NDCG / valid_user, HT / valid_user
