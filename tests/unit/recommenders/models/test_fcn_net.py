# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest

try:
    import torch
    import torch.nn as nn

    from recommenders.models.deeprec.models.pytorch.fcn_net import FcnNet
except ImportError:
    pass  # skip if torch is not installed


@pytest.fixture
def build_fcn_net():
    """Build an FcnNet whose weights are all 1 and biases all 0."""

    def build(**overrides):
        kwargs = dict(
            input_dim=3,
            layer_sizes=[2],
            activation=nn.ReLU(),
            dropout=[0.0],
            enable_BN=False,
            init_weight=nn.init.ones_,
        )
        kwargs.update(overrides)
        return FcnNet(**kwargs)

    return build


def test_output_shape(build_fcn_net):
    net = build_fcn_net(input_dim=5, layer_sizes=[8, 4], dropout=[0.0, 0.0])

    assert net(torch.randn(7, 5)).shape == (7, 1)


def test_matches_the_closed_form_stack(build_fcn_net):
    net = build_fcn_net().eval()

    # every weight is 1 and every bias 0, so the hidden layer sums the input and
    # the output layer sums the hidden units
    with torch.no_grad():
        logit = net(torch.tensor([[1.0, 2.0, 3.0]]))

    assert torch.allclose(logit, torch.tensor([[12.0]]))


def test_applies_the_activation(build_fcn_net):
    negative = build_fcn_net(init_weight=lambda w: nn.init.constant_(w, -1.0)).eval()
    x = torch.tensor([[1.0, 2.0, 3.0]])

    with torch.no_grad():
        relu_logit = negative(x)
        negative.activation = nn.Identity()
        identity_logit = negative(x)

    # pre-activations are -6, so ReLU zeroes them and the logit collapses to the
    # output bias, while identity carries them through
    assert torch.allclose(relu_logit, torch.tensor([[0.0]]))
    assert torch.allclose(identity_logit, torch.tensor([[12.0]]))


def test_initializes_every_linear_with_the_callable_and_zeroes_biases(build_fcn_net):
    net = build_fcn_net(
        layer_sizes=[4, 2],
        dropout=[0.0, 0.0],
        init_weight=lambda w: nn.init.constant_(w, 0.25),
    )

    for linear in (*net.linears, net.out):
        assert torch.all(linear.weight == 0.25)
        assert torch.all(linear.bias == 0.0)


@pytest.mark.parametrize(
    "enable_BN, expected", [(True, "BatchNorm1d"), (False, "Identity")]
)
def test_inserts_batch_norm_only_when_enabled(build_fcn_net, enable_BN, expected):
    net = build_fcn_net(enable_BN=enable_BN)

    assert type(net.bns[0]).__name__ == expected


def test_dropout_applies_in_training_and_not_in_eval(build_fcn_net):
    net = build_fcn_net(dropout=[1.0])
    x = torch.tensor([[1.0, 2.0, 3.0]])

    with torch.no_grad():
        train_logit = net.train()(x)
        eval_logit = net.eval()(x)

    # a rate of 1.0 drops every hidden unit while training, leaving the output bias
    assert torch.allclose(train_logit, torch.tensor([[0.0]]))
    assert torch.allclose(eval_logit, torch.tensor([[12.0]]))


def test_rejects_a_dropout_length_mismatch(build_fcn_net):
    with pytest.raises(ValueError, match="one rate per hidden layer"):
        build_fcn_net(layer_sizes=[4, 2], dropout=[0.0])
