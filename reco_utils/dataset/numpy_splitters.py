# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Collection of numpy based splitters

"""

import numpy as np


def numpy_stratified_split(X, ratio=0.75, seed=123):

    """
    Split the user/item affinity matrix into train and test set matrices while mantaining
    local (i.e. per user) ratios.

    Args:
        X (np.array, int): a sparse matrix
        ratio (scalar, float): fraction of the entire dataset to constitute the train set
        seed (scalar, int): random seed

    Returns:
        Xtr (np.array, int): train set user/item affinity matrix
        Xtst (np.array, int): test set user/item affinity matrix

    Basic mechanics:
        Main points :

        1. In a typical recommender problem, different users rate a different number of items,
           and therefore the user/affinity matrix has a sparse structure with variable number
           of zeroes (unrated items) per row (user). Cutting a total amount of ratings will
           result in a non-homogenou distribution between train and test set, i.e. some test
           users may have many ratings while other very little if none.

        2. In an unsupervised learning problem, no explicit answer is given. For this reason
           the split needs to be implemented in a different way then in supervised learningself.
           In the latter, one typically split the dataset by rows (by examples), ending up with
           the same number of feautures but different number of examples in the train/test setself.
           This scheme does not work in the unsupervised case, as part of the rated items needs to
           be used as a test set for fixed number of users.

        Solution:

        1. Instead of cutting a total percentage, for each user we cut a relative ratio of the rated
           items. For example, if user1 has rated 4 items and user2 10, cutting 25% will correspond to
           1 and 2.6 ratings in the test set, approximated as 1 and 3 according to the round() function.
           In this way, the 0.75 ratio is satified both locally and globally, preserving the original
           distribution of ratings across the train and test set.

        2. It is easy (and fast) to satisfy this requirements by creating the test via element subtraction
           from the original datatset X. We first create two copies of X; for each user we select a random
           sample of local size ratio (point 1) and erase the remaining ratings, obtaining in this way the
           train set matrix Xtst. The train set matrix is obtained in the opposite way.


    """

    np.random.seed(seed)  # set the random seed

    test_cut = int((1 - ratio) * 100)  # percentage of ratings to go in the test set

    # initialize train and test set matrices
    Xtr = X.copy()
    Xtst = X.copy()

    # find the number of rated movies per user
    rated = np.sum(Xtr != 0, axis=1)

    # for each user, cut down a test_size% for the test set
    tst = np.around((rated * test_cut) / 100).astype(int)

    Nusers, Nitems = X.shape  # total number of users and items

    for u in range(Nusers):
        # For each user obtain the index of rated movies
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()

        # extract a random subset of size n from the set of rated movies without repetition
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))

        Xtr[
            u, idx_tst
        ] = 0  # change the selected rated movies to unrated in the train set
        Xtst[
            u, idx_train
        ] = 0  # set the movies that appear already in the train set as 0

    del idx, idx_train, idx_tst

    return Xtr, Xtst
