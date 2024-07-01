# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/model/sequential/sasrec.py
#

import torch
import torch.nn as nn

from recommenders.models.unirec.model.sequential.seqrec_base import SeqRecBase
import recommenders.models.unirec.model.modules as modules


class SASRec(SeqRecBase):
    def __init__(self, config):
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = float(config["layer_norm_eps"])
        self.max_seq_len = config["max_seq_len"]
        self.use_pos_emb = config["use_position_emb"]
        super(SASRec, self).__init__(config)

    def _define_model_layers(self):
        # multi-head attention
        self.position_embedding = (
            nn.Embedding(self.max_seq_len + 1, self.hidden_size)
            if self.use_pos_emb
            else None
        )  # +1 for consistent with ranking model
        self.trm_encoder = modules.TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64

        if self.use_pos_emb:
            # mask for left-to-right unidirectional
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(
                torch.ones(attn_shape), diagonal=1
            )  # torch.uint8
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long().to(item_seq.device)

            extended_attention_mask = extended_attention_mask * subsequent_mask

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward_user_emb(
        self,
        user_id=None,
        item_seq=None,
        item_seq_len=None,
        item_seq_features=None,
        time_seq=None,
    ):
        item_emb = self.item_embedding_for_user(item_seq, item_seq_features, time_seq)
        input_emb = item_emb
        if self.position_embedding is not None:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = input_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self._get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = output[:, -1, :]
        return output  # [B H]
