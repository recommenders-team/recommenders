# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os

import pytest
import numpy as np

try:
    import torch

    from recommenders.models.rbm.rbm import RBM
except ImportError:
    pass  # skip these imports if we are in a cpu environment without torch


@pytest.fixture(scope="module")
def init_rbm():
    return {
        # model properties (constructor)
        "possible_ratings": [1, 2, 3, 4, 5],
        "n_visible": 500,
        "n_hidden": 100,
        "init_stdv": 0.01,
        "seed": 42,
        # training properties (fit)
        "training_epoch": 10,
        # must stay <= the number of users of the affinity_matrix fixture, otherwise no
        # minibatch is created and fit() raises
        "minibatch": 10,
        "keep_prob": 0.8,
        "learning_rate": 0.002,
        "sampling_protocol": [30, 50, 80, 90, 100],
        "display_epoch": 20,
    }


@pytest.mark.gpu
def test_class_init(init_rbm):
    # only the model properties are passed to the constructor
    model = RBM(
        possible_ratings=init_rbm["possible_ratings"],
        visible_units=init_rbm["n_visible"],
        hidden_units=init_rbm["n_hidden"],
        init_stdv=init_rbm["init_stdv"],
        seed=init_rbm["seed"],
    )

    # list of unique rating values
    assert np.array_equal(model.possible_ratings, init_rbm["possible_ratings"])
    # number of visible units
    assert model.n_visible == init_rbm["n_visible"]
    # number of hidden units
    assert model.n_hidden == init_rbm["n_hidden"]
    # standard deviation used to initialize the weight matrix
    assert model.stdv == init_rbm["init_stdv"]
    # seed
    assert model.seed == init_rbm["seed"]

    # the parameters are created at construction time
    assert tuple(model.w.shape) == (init_rbm["n_visible"], init_rbm["n_hidden"])
    assert tuple(model.bv.shape) == (1, init_rbm["n_visible"])
    assert tuple(model.bh.shape) == (1, init_rbm["n_hidden"])


@pytest.mark.gpu
def test_train_param_init(init_rbm, affinity_matrix):
    # obtain the train/test set matrices
    Xtr, _ = affinity_matrix

    # initialize the model with the model properties only
    model = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr), np.array([0])),
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
    )
    # fit the model to the data, passing the training properties
    model.fit(
        Xtr,
        training_epoch=init_rbm["training_epoch"],
        minibatch_size=init_rbm["minibatch"],
    )

    # weight matrix
    assert tuple(model.w.shape) == (Xtr.shape[1], init_rbm["n_hidden"])
    # bias, visible units
    assert tuple(model.bv.shape) == (1, Xtr.shape[1])
    # bias, hidden units
    assert tuple(model.bh.shape) == (1, init_rbm["n_hidden"])

    # the training hyperparameters are recorded on fit
    assert model.epochs == init_rbm["training_epoch"]
    assert model.minibatch == init_rbm["minibatch"]
    # one rmse value is collected per training epoch
    assert len(model.rmse_train) == init_rbm["training_epoch"]


@pytest.mark.gpu
def test_sampling_funct(init_rbm, affinity_matrix):
    # obtain the train/test set matrices
    Xtr, _ = affinity_matrix

    # initialize the model
    model = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr), np.array([0])),
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
    )

    def check_sampled_values(sampled, s):
        """
        Check if the elements of the sampled units are in {0,s}
        """
        a = []

        for i in range(0, s + 1):
            l_bool = sampled == i
            a.append(l_bool)

        return sum(a)

    r = int(Xtr.max())  # obtain the rating scale
    n_ratings = len(np.setdiff1d(np.unique(Xtr), np.array([0])))  # number of classes

    # fit the model to the data
    model.fit(
        Xtr,
        training_epoch=init_rbm["training_epoch"],
        minibatch_size=init_rbm["minibatch"],
    )
    model.eval()  # disable dropout so the sampled units are not masked out

    # clamp the visible units on the data
    v = torch.as_tensor(Xtr, dtype=torch.float32, device=model.device)

    # evaluate the activation probabilities of the hidden units and their sampled values
    with torch.no_grad():
        phv_t, h_t = model.sample_hidden_units(v)

    phv = phv_t.cpu().numpy()
    h = h_t.cpu().numpy()

    # check the dimensions of the two matrices
    assert phv.shape == (Xtr.shape[0], init_rbm["n_hidden"])
    assert h.shape == (Xtr.shape[0], init_rbm["n_hidden"])

    # check that the activation probabilities are in [0,1]
    assert (phv <= 1).all() & (phv >= 0).all()

    # check that the sampled value of the hidden units is either 1 or 0
    assert check_sampled_values(h, 1).all()

    # evaluate the activation probabilities of the visible units and their sampled values.
    # the original (clamped) visible units are passed so that unrated items stay masked.
    with torch.no_grad():
        pvh_t, v_sampled_t = model.sample_visible_units(h_t, v)

    pvh = pvh_t.cpu().numpy()
    v_sampled = v_sampled_t.cpu().numpy()

    assert pvh.shape == (Xtr.shape[0], Xtr.shape[1], n_ratings)
    assert v_sampled.shape == Xtr.shape

    # check that the multinomial distribution is normalized over the r classes for all users/items
    assert np.sum(pvh, axis=2) == pytest.approx(np.ones(Xtr.shape), rel=1e-5)

    # check that the sampled values of the visible units is in [0,r]
    assert check_sampled_values(v_sampled, r).all()


@pytest.mark.gpu
def test_gibbs_protocol(init_rbm, affinity_matrix):
    """The number of Gibbs sampling steps k is annealed according to the sampling
    protocol. A protocol that is fully consumed before the end of training must not
    raise an IndexError.
    """
    Xtr, _ = affinity_matrix

    model = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr), np.array([0])),
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
    )

    # a short protocol, deliberately not ending at 100, so that it is exhausted while
    # there are still epochs left to run
    sampling_protocol = [30, 50, 80]

    model.fit(
        Xtr,
        training_epoch=20,
        minibatch_size=init_rbm["minibatch"],
        sampling_protocol=sampling_protocol,
    )

    # k starts at 1 and is incremented once per protocol transition
    assert model.k > 1
    assert model.k <= len(sampling_protocol)
    # the protocol index never runs past the last usable entry
    assert model.l <= len(sampling_protocol) - 1


@pytest.mark.gpu
def test_fit_raises_when_minibatch_larger_than_train_set(init_rbm, affinity_matrix):
    """A minibatch larger than the train set would produce zero minibatches per epoch
    and silently leave the model untrained, so fit() must fail fast.
    """
    Xtr, _ = affinity_matrix

    model = RBM(
        possible_ratings=np.setdiff1d(np.unique(Xtr), np.array([0])),
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
    )

    with pytest.raises(ValueError):
        model.fit(
            Xtr,
            training_epoch=init_rbm["training_epoch"],
            minibatch_size=Xtr.shape[0] + 1,
        )


@pytest.mark.gpu
def test_save_load(init_rbm, affinity_matrix, tmp_path):

    # obtain the train/test set matrices
    Xtr, _ = affinity_matrix

    possible_ratings = np.setdiff1d(np.unique(Xtr), np.array([0]))

    # initialize and train the model
    original_model = RBM(
        possible_ratings=possible_ratings,
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
        init_stdv=init_rbm["init_stdv"],
        seed=init_rbm["seed"],
    )
    original_model.fit(
        Xtr,
        training_epoch=init_rbm["training_epoch"],
        minibatch_size=init_rbm["minibatch"],
        keep_prob=init_rbm["keep_prob"],
        learning_rate=init_rbm["learning_rate"],
        sampling_protocol=init_rbm["sampling_protocol"],
        display_epoch=init_rbm["display_epoch"],
    )

    # save the model
    file_path = os.path.join(tmp_path, "rbm_model.pt")
    original_model.save(file_path)

    # initialize another model with the same architecture
    saved_model = RBM(
        possible_ratings=possible_ratings,
        visible_units=Xtr.shape[1],
        hidden_units=init_rbm["n_hidden"],
        init_stdv=init_rbm["init_stdv"],
        seed=init_rbm["seed"],
    )

    # load the pretrained model
    saved_model.load(file_path)

    # list of unique rating values
    assert np.array_equal(saved_model.possible_ratings, original_model.possible_ratings)
    # number of visible units
    assert saved_model.n_visible == original_model.n_visible
    # number of hidden units
    assert saved_model.n_hidden == original_model.n_hidden
    # standard deviation used to initialize the weight matrix
    assert saved_model.stdv == original_model.stdv

    # the learned parameters are restored
    assert torch.allclose(saved_model.w, original_model.w)
    assert torch.allclose(saved_model.bv, original_model.bv)
    assert torch.allclose(saved_model.bh, original_model.bh)
