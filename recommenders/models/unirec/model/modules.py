# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import copy
import math
from typing import *
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np

from unirec.constants.global_variables import *


def bpr_loss(pos_score, neg_score, reduction=True):
    loss = -torch.log(EPS + torch.sigmoid(pos_score - neg_score))
    if reduction:
        loss = loss.mean()
    else:
        loss = loss.mean(dim=-1)
    return loss


r"""
    CCL is cosine contrastive loss, proposed by 
    SimpleX: A Simple and Strong Baseline for Collaborative Filtering. CIKM'21.
    https://dl.acm.org/doi/10.1145/3459637.3482297
"""


def ccl_loss(pos_score, neg_score, w, m, reduction=False):
    loss = (
        1
        - pos_score
        + w * torch.mean(torch.clamp(neg_score - m, min=0), dim=-1, keepdim=False)
    )
    if reduction:
        loss = loss.mean()
    return loss


def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))


def softmax_np(x, axis=-1):
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


class InnerProductScorer(nn.Module):
    def __init__(self):
        super(InnerProductScorer, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        assert (
            math.fabs(x.dim() - y.dim()) <= 1
        ), "difference of dimension between x and y should be no more than 1."
        # 3 cases. Some cases could be merged but not easy to understanding after merge
        ## case1: x.dim() == y.dim(), e.g. [N,D]x[N,D]-one user vs one item, [N,D]x[M,D]-each user vs all items,  ([B,N,D]x[B,N,D], [B,N,D]x[B,M,D], optional, not needed now)
        if x.dim() == y.dim():
            if x.size(0) == y.size(0):  # one user vs one item
                res = (x * y).sum(-1)  # [N,]
            else:  # each user vs all items
                res = torch.matmul(x, y.transpose(-1, -2))  # [N, M]
        ## case2: x.dim() = y.dim()+1, e.g. [N,D]x[D], [B,N,D]x[B,D] , one item vs multiple users
        elif x.dim() > y.dim():
            if y.dim() == 1:
                res = torch.matmul(x, y)  # [N,]
            else:
                res = torch.matmul(x, y.view(*y.shape, 1)).squeeze(-1)  # [B, N]
        ## case3: x.dim() = y.dim()-1, e.g. [D]x[N,D], [B,D]x[B,N,D], one user vs multiple items
        else:
            res = self.forward(y, x)
        return res


class CosineScorer(InnerProductScorer):
    def __init__(self, eps=1e-6):
        super(CosineScorer, self).__init__()
        self.eps = nn.parameter.Parameter(
            torch.tensor(eps).type(torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_length = (x * x).sum(-1, keepdim=True)
        y_length = (y * y).sum(-1, keepdim=True)
        deno = super().forward(x_length, y_length)
        ip_score = super().forward(x, y)
        res = ip_score / torch.maximum(deno, self.eps)
        return res


class MLPScorer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Union[int, List[int]],
        dropout_prob: float,
        act_f: str = "tanh",
    ):
        super(MLPScorer, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self.act_f = act_f

        seq = []
        if type(hidden_dim) == int:
            hidden_dim = [2 * self.embed_dim, hidden_dim]
        else:  # list
            hidden_dim = [2 * self.embed_dim] + hidden_dim
        for n_in, n_out in zip(hidden_dim[:-1], hidden_dim[1:]):
            seq.append(nn.Dropout(p=dropout_prob))
            seq.append(nn.Linear(n_in, n_out))
            seq.append(self._get_act_func(self.act_f))
        seq.append(nn.Linear(hidden_dim[-1], 1))
        self.model = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        assert (
            math.fabs(x.dim() - y.dim()) <= 1
        ), "difference of dimension between x and y should be no more than 1."
        if x.dim() == y.dim():  # e.g. [N,D]x[N,D], [N,D]x[M,D]
            if x.size(0) == y.size(0):
                pass
            else:
                x = x.view(x.size(0), 1, -1).expand(
                    -1, y.size(0), -1
                )  # [N, N, D] or [N, M, D]
                y = y.view(-1, *y.shape).expand(
                    x.size(0), -1, -1
                )  # # [N, N, D] or [N, M, D]
        elif x.dim() > y.dim():  # e.g. [N,D]x[D], [B,N,D]x[B,D]
            y = y.unsqueeze(-2).expand_as(x)
        elif x.dim() < y.dim():  # e.g. [D]x[N,D], [B,D]x[B,N,D]
            x = x.unsqueeze(-2).expand_as(y)
        input = torch.cat([x, y], -1)  # [N,2D] or [N,M,2D]
        res = self.model(input).squeeze(-1)  # [N] or [N,M]
        return res

    def _get_act_func(self, func_name: str):
        name = func_name.lower()
        if name == "sigmoid":
            f = nn.Sigmoid()
        elif name == "relu":
            f = nn.ReLU()
        elif name == "tanh":
            f = nn.Tanh()
        elif name == "leakyrelu":
            f = nn.LeakyReLU()  # negative_slope parameter set as default now
        else:
            raise NotImplementedError(
                f"Activation function {func_name} not supported now."
            )
        return f


class Dice(nn.Module):
    r"""Dice activation function
    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s
    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(self, mask_mat, input_size=64, output_size=64):
        super(SequenceAttLayer, self).__init__()

        self.mask_mat = mask_mat
        self.dense_1 = nn.Linear(input_size, output_size, bias=False)
        self.dense_2 = nn.Linear(input_size, output_size, bias=False)

        # for activation_unit
        # self.activation = nn.ReLU()
        # self.dense = nn.Linear(input_size*4, 1)
        # self.activation = Dice(100)

    def activation_unit(self, queries, keys):
        """
        to get activation weight: [2048, 5, 100]
        """
        batch_size = queries.shape[0]
        test_items_num = queries.shape[1]
        seq_len = keys.shape[1]
        embedding_size = queries.shape[-1]
        queries_expand_emb = queries.unsqueeze(2).expand(
            batch_size, test_items_num, seq_len, embedding_size
        )  # [2048, 5, 100, 64]
        keys_expand_emb = keys.unsqueeze(1).expand(
            batch_size, test_items_num, seq_len, embedding_size
        )  # [2048, 5, 100, 64]

        concat_emb = torch.cat(
            [
                queries_expand_emb,
                keys_expand_emb,
                queries_expand_emb - keys_expand_emb,
                queries_expand_emb * keys_expand_emb,
            ],
            dim=-1,
        )
        output = self.dense(concat_emb).squeeze(-1)
        # output = self.activation(output)

        return output

    def forward(self, queries, keys, keys_length):
        """
        queries: [2048, 5, 64]
        keys: [2048, 100, 64]
        """
        embedding_size = queries.shape[-1]  # H

        new_queries = self.dense_1(queries)
        new_keys = self.dense_2(keys)

        att_scores = torch.matmul(
            new_queries, new_keys.transpose(-1, -2)
        )  # [2048, 5, 100]
        # att_scores = self.activation_unit(new_queries, new_keys)

        # get mask
        mask = self.mask_mat.repeat(new_queries.size(0), 1)  # [2048, 100]
        saved_idx_thre = self.mask_mat.shape[-1] - keys_length
        mask = mask < saved_idx_thre.unsqueeze(1)  # [2048, 100]
        mask = mask.unsqueeze(1).expand_as(att_scores)
        mask = mask.reshape(
            new_queries.shape[0], -1, new_keys.shape[1]
        )  # [2048, 5, 100]

        mask_value = 0.0

        att_scores = att_scores.masked_fill(mask=mask, value=torch.tensor(mask_value))
        att_scores = att_scores / (embedding_size**0.5)
        att_scores = nn.Softmax(dim=-1)(att_scores)
        output = torch.matmul(att_scores, keys)  # [2048, 5, 64]

        return output


class AttentionMergeLayer(nn.Module):
    def __init__(self, input_size, dropout):
        super(AttentionMergeLayer, self).__init__()
        self.input_size = input_size
        self.dense = nn.Linear(self.input_size, self.input_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.randn([self.input_size, 1]))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, item_seq_emb):
        item_seq_emb = self.dense(item_seq_emb)
        att_scores = torch.matmul(item_seq_emb, self.h).squeeze(-1)
        att_scores = self.softmax(att_scores)

        att_emb = torch.matmul(
            att_scores.unsqueeze(-1).transpose(-1, -2), item_seq_emb
        ).squeeze(1)
        att_emb = self.emb_dropout(att_emb)

        return att_emb  # [B, embedding_size]


# Transformer
class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": nn.GELU(
                approximate="none"
            ),  # approximate can be set to 'tanh' for OpenAI GPT's gelu: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return ACT2FN[act]

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.
    - n_layers(num): num of transformer layers in transformer encoder. Default: 2
    - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - hidden_size(num): the input and output hidden size. Default: 64
    - inner_size(num): the dimensionality in feed-forward layer. Default: 256
    - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
    - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                  candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


r"""
The NeuProcessEncoder class is an implementation of the Neural Process (NP) encoder described in the AdaRanker paper. 

The NP encoder is used to approximate a stochastic process with learnable neural networks to model data distributions associated with a ranking request. It generates a latent embedding vector for each item in the candidate set using a two-layer MLP, then aggregates these latent vectors to generate a permutation-invariant representation via mean pooling. 

This representation is then used to generate the mean vector and the variance vector. The data distribution is modeled by a random variable z, which is implemented with the reparameterization trick, making all the computational nodes in the model differentiable and gradients can be smoothly backpropagated. 

The encoder is divided into two parts, one for encoding item embeddings and the other for encoding the latent vector z.
"""


class NeuProcessEncoder(nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=64,
        output_size=64,
        dropout_prob=0.4,
        device=None,
    ):
        super(NeuProcessEncoder, self).__init__()
        self.device = device

        # Encoder for item embeddings
        layers = [
            nn.Linear(input_size, hidden_size),
            torch.nn.Dropout(dropout_prob),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
        ]
        self.input_to_hidden = nn.Sequential(*layers)

        # Encoder for latent vector z
        self.z1_dim = input_size  # 64
        self.z2_dim = hidden_size  # 64
        self.z_dim = output_size  # 64
        self.z_to_hidden = nn.Linear(self.z1_dim, self.z2_dim)
        self.hidden_to_mu = nn.Linear(self.z2_dim, self.z_dim)
        self.hidden_to_logsigma = nn.Linear(self.z2_dim, self.z_dim)

    def emb_encode(self, input_tensor):
        hidden = self.input_to_hidden(input_tensor)

        return hidden

    def aggregate(self, input_tensor):
        return torch.mean(input_tensor, dim=-2)

    def z_encode(self, input_tensor):
        hidden = torch.relu(self.z_to_hidden(input_tensor))
        mu = self.hidden_to_mu(hidden)
        log_sigma = self.hidden_to_logsigma(hidden)
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, log_sigma

    def encoder(self, input_tensor):
        z_ = self.emb_encode(input_tensor)
        z = self.aggregate(z_)
        self.z, mu, log_sigma = self.z_encode(z)
        return self.z, mu, log_sigma

    def forward(self, input_tensor):
        self.z, _, _ = self.encoder(input_tensor)
        return self.z


r"""
The AdaLinear class is an implementation of the Ada-Ranker model described in the AdaRanker paper. 

Ada-Ranker is designed to modulate the latent representations of input sequences, which can be shared by different sequential recommendation models. This class provides a method to learn to generate two modulation coefficients, 𝛾 and 𝛽, through the conditional representation z. These coefficients are then used to adjust the latent representations of the item sequence.

This class is also responsible for modulating the parameters of the scoring function g^{PRED}(·) using the idea of model patch. This involves generating parameter patches for the 𝑘-th hidden layer of the predictive layer according to the conditional representation z, then modulating the ranker's MLP parameters.
"""


class AdaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(AdaLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight_new.transpose(-1, -2))

        output = input.matmul(self.weight_new.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias_new
        return output

    def adaptive_parameters(self, batch_size, gama, beta):
        """
        gama: [batch_size, self.out_features, self.in_features]
        beta: [batch_size, 1]
        self.weight.data: [self.out_features, self.in_features]
        """
        gama_w = gama.unsqueeze(1).expand(
            [batch_size, self.out_features, self.in_features]
        )
        beta_w = beta.unsqueeze(1)
        gama_b = gama.expand([batch_size, self.out_features])
        beta_b = beta

        self.weight_specific = (
            self.weight * gama_w + beta_w
        )  # [batch_size, self.out_features, self.in_features]
        self.weight_new = self.weight_specific * self.weight

        if self.bias is not None:
            self.bias_specific = self.bias * gama_b + beta_b
            self.bias_new = self.bias_specific + self.bias
            self.bias_new = self.bias_new.unsqueeze(1)

    def adaptive_parameters_ws(self, batch_size, gama, beta):
        """
        gama: [batch_size, self.out_features, self.in_features]
        beta: [batch_size, 1]
        self.weight.data: [self.out_features, self.in_features]
        """
        gama_w = gama.unsqueeze(1).expand(
            [batch_size, self.out_features, self.in_features]
        )
        beta_w = beta.unsqueeze(1)
        gama_b = gama.expand([batch_size, self.out_features])
        beta_b = beta

        self.weight_new = (
            self.weight * gama_w + beta_w
        )  # [batch_size, self.out_features, self.in_features]

        if self.bias is not None:
            self.bias_new = self.bias * gama_b + beta_b
            self.bias_new = self.bias_new.unsqueeze(1)

    def memory_parameters(self, mem_wei, mem_bias):
        self.weight_specific = (
            mem_wei  # [batch_size, self.out_features, self.in_features]
        )
        self.weight_new = self.weight_specific * self.weight

        if self.bias is not None:
            self.bias_specific = mem_bias.squeeze(-1)
            self.bias_new = self.bias_specific + self.bias
            self.bias_new = self.bias_new.unsqueeze(1)

    def add_bias_only(self, bias_vec):
        self.weight_new = self.weight
        self.bias_new = bias_vec + self.bias
        self.bias_new = self.bias_new.unsqueeze(1)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


r"""
The MemoryUnit class implements a parameter memory network with L base parameters. 

This network stores multiple base parameters and uses a linear combination of them to generate parameter patches. The coefficients for this linear combination are determined by a set of reading heads. 

This class also provides a method for calculating the regularization loss for the memory unit.
"""


class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, clusters_k=10):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.input_size = input_size
        self.output_size = output_size
        self.array = nn.Parameter(
            init.xavier_uniform_(
                torch.FloatTensor(self.clusters_k, input_size * output_size)
            )
        )
        self.index = nn.Parameter(
            init.xavier_uniform_(torch.FloatTensor(self.clusters_k, emb_size))
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        att_scores = torch.matmul(
            bias_emb, self.index.transpose(-1, -2)
        )  # [batch_size, clusters_k]
        att_scores = self.softmax(att_scores)

        # [batch_size, input_size, output_size]
        para_new = torch.matmul(
            att_scores, self.array
        )  # [batch_size, input_size*output_size]
        para_new = para_new.view(-1, self.output_size, self.input_size)

        return para_new

    def reg_loss(self, reg_weights=1e-2):
        loss_1 = reg_weights * self.array.norm(2)
        loss_2 = reg_weights * self.index.norm(2)

        return loss_1 + loss_2


class ModulateHidden(nn.Module):
    def __init__(self, input_size, emb_size):
        super(ModulateHidden, self).__init__()
        self.input_size = input_size
        self.emb_size = emb_size
        self.gen_para_layer = nn.Linear(
            self.emb_size, self.input_size * self.input_size
        )

    def gen_para(self, bias_emb):
        """
        bias_emb: [batch_size, emb_size]
        """
        para_new = self.gen_para_layer(
            bias_emb
        )  # [batch_size, self.input_size*self.output_size]
        self.para_new = para_new.view(-1, self.input_size, self.input_size)

    def forward(self, input: torch.Tensor):
        output = input.matmul(self.para_new.transpose(-1, -2))

        return output


class AdapLinear_mmoe(nn.Module):
    def __init__(
        self,
        config,
        emb_size,
        in_features: int,
        out_features: int,
        bias: bool = True,
        expert_num=10,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(AdapLinear_mmoe, self).__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        _weight = torch.randn(
            (expert_num, self.out_features * self.in_features), requires_grad=True
        ).to(self.device)
        self.weight = nn.Parameter(_weight)

        if bias:
            _bias = torch.randn((expert_num, self.out_features), requires_grad=True).to(
                self.device
            )
            self.bias = nn.Parameter(_bias)
            # self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))

        # gate
        self.gate_net = nn.Linear(emb_size, expert_num, bias=False)
        self.softmax = nn.Softmax(-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        if input.dim() == 2 and self.bias is not None:
            return torch.addmm(self.bias, input, self.weight_new.transpose(-1, -2))

        output = input.matmul(self.weight_new.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias_new
        return output

    def adaptive_parameters(self, domain_bias):
        # domain_bias: [batch_size, emb_size]
        if len(domain_bias.size()) == 3:
            domain_bias = domain_bias.squeeze(1)
        att_scores = self.gate_net(domain_bias)  # [batch_size, expert_num]
        att_scores = self.softmax(att_scores)
        self.weight_new = torch.matmul(
            att_scores, self.weight
        )  # [batch_size, input_size*output_size]
        self.weight_new = self.weight_new.view(
            -1, self.out_features, self.in_features
        )  # [batch_size, self.out_features, self.in_features]
        if self.bias is not None:
            self.bias_new = torch.matmul(att_scores, self.bias).unsqueeze(
                1
            )  # [batch_size, input_size*output_size]

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class MMoEUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, input_size, output_size, emb_size, expert_num=10):
        super(MMoEUnit, self).__init__()
        self.expert_num = expert_num
        self.input_size = input_size
        self.output_size = output_size

        _weight = torch.randn(
            (expert_num, self.output_size * self.input_size), requires_grad=True
        )
        self.weight = nn.Parameter(_weight)

        # gate
        self.gate_net = nn.Linear(emb_size, expert_num, bias=False)
        self.softmax = nn.Softmax(-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, bias_emb):
        """
        bias_emb: [batch_size, 1, emb_size]
        """
        if len(bias_emb.size()) == 3:
            bias_emb = bias_emb.squeeze(1)
        att_scores = self.gate_net(bias_emb)  # [batch_size, expert_num]
        att_scores = self.softmax(att_scores)
        para_new = torch.matmul(
            att_scores, self.weight
        )  # [batch_size, input_size*output_size]
        para_new = para_new.view(
            -1, self.output_size, self.input_size
        )  # [batch_size, self.out_features, self.in_features]

        return para_new
