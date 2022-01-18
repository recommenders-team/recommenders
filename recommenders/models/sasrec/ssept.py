# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import tensorflow as tf
from recommenders.models.sasrec.model import SASREC, Encoder, LayerNormalization


class SSEPT(SASREC):
    """
    SSE-PT Model

    :Citation:

    Wu L., Li S., Hsieh C-J., Sharpnack J., SSE-PT: Sequential Recommendation
    Via Personalized Transformer, RecSys, 2020.
    TF 1.x codebase: https://github.com/SSE-PT/SSE-PT
    TF 2.x codebase (SASREc): https://github.com/nnkkmto/SASRec-tf2

    Args:
        basic arguments -
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of Transformer blocks to be used
        embedding_dim: item embedding dimension
        attention_dim: Transformer attention dimension
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing

        additional arguments -
        user_num: number of users in the dataset
        user_embedding_dim: user embedding dimension
        item_embedding_dim: item embedding dimension

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.user_num = kwargs.get("user_num", None)  # New
        self.conv_dims = kwargs.get("conv_dims", [200, 200])  # modified
        self.user_embedding_dim = kwargs.get(
            "user_embedding_dim", self.embedding_dim
        )  # extra
        self.item_embedding_dim = kwargs.get("item_embedding_dim", self.embedding_dim)
        self.hidden_units = self.item_embedding_dim + self.user_embedding_dim

        # New, user embedding
        self.user_embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.user_num + 1,
            output_dim=self.user_embedding_dim,
            name="user_embeddings",
            mask_zero=True,
            input_length=1,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.positional_embedding_layer = tf.keras.layers.Embedding(
            self.seq_max_len,
            self.user_embedding_dim + self.item_embedding_dim,  # difference
            name="positional_embeddings",
            mask_zero=False,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout_rate)
        self.encoder = Encoder(
            self.num_blocks,
            self.seq_max_len,
            self.hidden_units,
            self.hidden_units,
            self.attention_num_heads,
            self.conv_dims,
            self.dropout_rate,
        )
        self.mask_layer = tf.keras.layers.Masking(mask_value=0)
        self.layer_normalization = LayerNormalization(
            self.seq_max_len, self.hidden_units, 1e-08
        )

    def call(self, x, training):

        users = x["users"]
        input_seq = x["input_seq"]
        pos = x["positive"]
        neg = x["negative"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)

        # User Encoding
        # u0_latent = self.user_embedding_layer(users[0])
        # u0_latent = u0_latent * (self.embedding_dim ** 0.5)
        u_latent = self.user_embedding_layer(users)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        # return users

        # replicate the user embedding for all the items
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])  # (b, s, h)

        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [tf.shape(input_seq)[0], -1, self.hidden_units],
        )
        seq_embeddings += positional_embeddings

        # dropout
        seq_embeddings = self.dropout_layer(seq_embeddings, training=training)

        # masking
        seq_embeddings *= mask

        # --- ATTENTION BLOCKS ---
        seq_attention = seq_embeddings  # (b, s, h1 + h2)

        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)

        # --- PREDICTION LAYER ---
        # user's sequence embedding
        pos = self.mask_layer(pos)
        neg = self.mask_layer(neg)

        user_emb = tf.reshape(
            u_latent,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.user_embedding_dim],
        )
        pos = tf.reshape(pos, [tf.shape(input_seq)[0] * self.seq_max_len])
        neg = tf.reshape(neg, [tf.shape(input_seq)[0] * self.seq_max_len])
        pos_emb = self.item_embedding_layer(pos)
        neg_emb = self.item_embedding_layer(neg)

        # Add user embeddings
        pos_emb = tf.reshape(tf.concat([pos_emb, user_emb], 1), [-1, self.hidden_units])
        neg_emb = tf.reshape(tf.concat([neg_emb, user_emb], 1), [-1, self.hidden_units])

        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
        )  # (b*s, d)

        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        pos_logits = tf.expand_dims(pos_logits, axis=-1)  # (bs, 1)
        # pos_prob = tf.keras.layers.Dense(1, activation='sigmoid')(pos_logits)  # (bs, 1)

        neg_logits = tf.expand_dims(neg_logits, axis=-1)  # (bs, 1)
        # neg_prob = tf.keras.layers.Dense(1, activation='sigmoid')(neg_logits)  # (bs, 1)

        # output = tf.concat([pos_logits, neg_logits], axis=0)

        # masking for loss calculation
        istarget = tf.reshape(
            tf.cast(tf.not_equal(pos, 0), dtype=tf.float32),
            [tf.shape(input_seq)[0] * self.seq_max_len],
        )

        return pos_logits, neg_logits, istarget

    def predict(self, inputs):
        """
        Model prediction for candidate (negative) items

        """
        training = False
        user = inputs["user"]
        input_seq = inputs["input_seq"]
        candidate = inputs["candidate"]

        mask = tf.expand_dims(tf.cast(tf.not_equal(input_seq, 0), tf.float32), -1)
        seq_embeddings, positional_embeddings = self.embedding(input_seq)  # (1, s, h)

        u0_latent = self.user_embedding_layer(user)
        u0_latent = u0_latent * (self.user_embedding_dim ** 0.5)  # (1, 1, h)
        u0_latent = tf.squeeze(u0_latent, axis=0)  # (1, h)
        test_user_emb = tf.tile(u0_latent, [1 + self.num_neg_test, 1])  # (101, h)

        u_latent = self.user_embedding_layer(user)
        u_latent = u_latent * (self.user_embedding_dim ** 0.5)  # (b, 1, h)
        u_latent = tf.tile(u_latent, [1, tf.shape(input_seq)[1], 1])  # (b, s, h)

        seq_embeddings = tf.reshape(
            tf.concat([seq_embeddings, u_latent], 2),
            [tf.shape(input_seq)[0], -1, self.hidden_units],
        )
        seq_embeddings += positional_embeddings  # (b, s, h1 + h2)

        seq_embeddings *= mask
        seq_attention = seq_embeddings
        seq_attention = self.encoder(seq_attention, training, mask)
        seq_attention = self.layer_normalization(seq_attention)  # (b, s, h1+h2)
        seq_emb = tf.reshape(
            seq_attention,
            [tf.shape(input_seq)[0] * self.seq_max_len, self.hidden_units],
        )  # (b*s1, h1+h2)

        candidate_emb = self.item_embedding_layer(candidate)  # (b, s2, h2)
        candidate_emb = tf.squeeze(candidate_emb, axis=0)  # (s2, h2)
        candidate_emb = tf.reshape(
            tf.concat([candidate_emb, test_user_emb], 1), [-1, self.hidden_units]
        )  # (b*s2, h1+h2)

        candidate_emb = tf.transpose(candidate_emb, perm=[1, 0])  # (h1+h2, b*s2)
        test_logits = tf.matmul(seq_emb, candidate_emb)  # (b*s1, b*s2)

        test_logits = tf.reshape(
            test_logits,
            [tf.shape(input_seq)[0], self.seq_max_len, 1 + self.num_neg_test],
        )  # (1, s, 101)
        test_logits = test_logits[:, -1, :]  # (1, 101)
        return test_logits

    def loss_function(self, pos_logits, neg_logits, istarget):
        """
        Losses are calculated separately for the positive and negative
        items based on the corresponding logits. A mask is included to
        take care of the zero items (added for padding).
        """

        pos_logits = pos_logits[:, 0]
        neg_logits = neg_logits[:, 0]

        # ignore padding items (0)
        # istarget = tf.reshape(
        #     tf.cast(tf.not_equal(self.pos, 0), dtype=tf.float32),
        #     [tf.shape(self.input_seq)[0] * self.seq_max_len],
        # )
        # for logits
        loss = tf.reduce_sum(
            -tf.math.log(tf.math.sigmoid(pos_logits) + 1e-24) * istarget
            - tf.math.log(1 - tf.math.sigmoid(neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        # for probabilities
        # loss = tf.reduce_sum(
        #         - tf.math.log(pos_logits + 1e-24) * istarget -
        #         tf.math.log(1 - neg_logits + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        reg_loss = tf.compat.v1.losses.get_regularization_loss()
        # reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        # loss += sum(reg_losses)
        loss += reg_loss

        return loss
