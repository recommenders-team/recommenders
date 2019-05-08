# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import os
import logging
from reco_utils.recommender.deep_autoencoder.utils import activation

log = logging.getLogger(__name__)


class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes, nl_type='selu', is_constrained=True,
                 dp_drop_prob=0.0, last_layer_activations=True):
        """
        Describes an AutoEncoder model

        based on: https://github.com/NVIDIA/DeepRecommender/

        :param layer_sizes: Encoder network description. Should start with
                            feature size (e.g. dimensionality of x).
        For example: [10000, 1024, 512] will result in:
          - encoder 2 layers: 10000x1024 and 1024x512. Representation layer
                              (z) will be 512
          - decoder 2 layers: 512x1024 and 1024x10000.
        :param nl_type: Type of no-linearity
        :param is_constrained: Should constrain decoder weights
        :param dp_drop_prob: Dropout drop probability
        :param last_layer_activations: Whether to apply activations on last
                                       decoder layer
        """
        super(AutoEncoder, self).__init__()
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        if dp_drop_prob > 0:
            self.drop = nn.Dropout(dp_drop_prob)
        self._last = len(layer_sizes) - 2
        self._nl_type = nl_type
        self.encode_w = nn.ParameterList(
            [nn.Parameter(torch.rand(layer_sizes[i + 1], layer_sizes[i]))
             for i in range(len(layer_sizes) - 1)])
        for ind, w in enumerate(self.encode_w):
            weight_init.xavier_uniform(w)

        self.encode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(layer_sizes[i + 1]))
             for i in range(len(layer_sizes) - 1)])

        reversed_enc_layers = list(reversed(layer_sizes))

        self.is_constrained = is_constrained
        if not is_constrained:
            self.decode_w = nn.ParameterList(
                [nn.Parameter(torch.rand(reversed_enc_layers[i + 1],
                                         reversed_enc_layers[i]))
                 for i in range(len(reversed_enc_layers) - 1)])
            for ind, w in enumerate(self.decode_w):
                weight_init.xavier_uniform(w)
        self.decode_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(reversed_enc_layers[i + 1]))
             for i in range(len(reversed_enc_layers) - 1)])

        log.debug("******************************")
        log.debug("******************************")
        log.debug(layer_sizes)
        log.debug("Dropout drop probability: {}".format(self._dp_drop_prob))
        log.debug("Encoder pass:")
        for ind, w in enumerate(self.encode_w):
            log.debug(w.data.size())
            log.debug(self.encode_b[ind].size())
        log.debug("Decoder pass:")
        if self.is_constrained:
            log.debug('Decoder is constrained')
            for ind, w in enumerate(list(reversed(self.encode_w))):
                log.debug(w.transpose(0, 1).size())
                log.debug(self.decode_b[ind].size())
        else:
            for ind, w in enumerate(self.decode_w):
                log.debug(w.data.size())
                log.debug(self.decode_b[ind].size())
        log.debug("******************************")
        log.debug("******************************")

    def encode(self, x):
        for ind, w in enumerate(self.encode_w):
            x = activation(input=F.linear(input=x,
                                          weight=w,
                                          bias=self.encode_b[ind]),
                           kind=self._nl_type)
        if self._dp_drop_prob > 0:  # apply dropout only on code layer
            x = self.drop(x)
        return x

    def decode(self, z):
        if self.is_constrained:
            # constrained autoencode re-uses weights from encoder
            for ind, w in enumerate(list(reversed(self.encode_w))):
                z = activation(input=F.linear(input=z,
                                              weight=w.transpose(0, 1),
                                              bias=self.decode_b[ind]),
                               # last layer or decoder should not apply
                               # non linearities
                               kind=(self._nl_type
                                     if ind != self._last
                                     or self._last_layer_activations
                                     else 'none'))
        else:
            for ind, w in enumerate(self.decode_w):
                z = activation(input=F.linear(input=z,
                                              weight=w,
                                              bias=self.decode_b[ind]),
                               # last layer or decoder should not apply
                               # non linearities
                               kind=(self._nl_type
                                     if ind != self._last
                                     or self._last_layer_activations
                                     else 'none'))
        return z

    def forward(self, x):
        return self.decode(self.encode(x))
