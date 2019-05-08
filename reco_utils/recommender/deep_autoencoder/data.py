# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os import listdir, path
from random import shuffle
import torch


class UserItemRecDataProvider:
    """User-Item data class
    based on: https://github.com/NVIDIA/DeepRecommender/
    """

    def __init__(self, params, user_id_map=None, item_id_map=None):
        """Initialize internal parameters and generates self.data.
        self.data is a dictionary that relates user, item and rating in an
        aggregated fashion, example: if the first column (here called major
        column) is the user, it stores it as the dictionary key and. Then for
        each user, it stores a list of item-rating pairs.
        :param params: Parameters of the recommender
        :param user_id_map: Mapping between input user data and internal user
                            representation.
        :param item_id_map: Mapping between input item data and internal item
                            representation.
        """
        # Initialize parameters
        self._params = params
        self._batch_size = self.params['batch_size']

        # Parameters related with the column order of the input data.
        # self._i_id is the index of the item, self._u_id is the index of the
        # user/customer and self._r_id is the index of the rating. The param
        # self._major defines if item or user are the first index (first
        # column) in the input data.
        self._i_id = (0 if 'itemIdInd' not in self.params
                      else self.params['itemIdInd'])
        self._u_id = (1 if 'userIdInd' not in self.params
                      else self.params['userIdInd'])
        self._r_id = (2 if 'ratingInd' not in self.params
                      else self.params['ratingInd'])
        self._major = ('items' if 'major' not in self.params
                       else self.params['major'])
        if not (self._major == 'items' or self._major == 'users'):
            raise ValueError(
                "Major must be 'users' or 'items', but got {}"
                .format(self._major))
        self._major_ind = self._i_id if self._major == 'items' else self._u_id
        self._minor_ind = self._u_id if self._major == 'items' else self._i_id

        # Input file, extension and delimiter
        self._src_file = params['src_file']
        self._extension = (".txt" if 'extension' not in self.params
                           else self.params['extension'])
        self._delimiter = ('\t' if 'delimiter' not in self.params
                           else self.params['delimiter'])
        self._header = (True if 'header' not in self.params
                        else self.params['header'])
        src_files = [self._src_file]

        # Initializes self._user_id_map and self._item_id_map, which are
        # mappings between the original input user and item values and an
        # internal representation.
        if user_id_map is None or item_id_map is None:
            self._build_maps()
        else:
            self._user_id_map = user_id_map
            self._item_id_map = item_id_map
        major_map = (self._item_id_map if self._major == 'items'
                     else self._user_id_map)
        minor_map = (self._user_id_map if self._major == 'items'
                     else self._item_id_map)
        self._vector_dim = len(minor_map)

        # Loop to read and format the data
        self.data = dict()
        error = 0
        for source_file in src_files:
            head = False
            with open(source_file, 'r') as src:
                for line in src.readlines():
                    # Read line
                    if self._header and not head:
                        head = True
                        continue
                    parts = line.strip().split(self._delimiter)
                    if len(parts) < 3:
                        raise ValueError(
                            'Encountered badly formatted line in {}'
                            .format(source_file))

                    # Remove non-seen values in case they exist
                    if int(parts[self._minor_ind]) not in minor_map:
                        # print("WARNING: value {} not seen in mapping".
                        # format(int(parts[self._minor_ind])))
                        error += 1
                        continue

                    # Parse the input data using the user and item mappings.
                    # The key of self.data uses the major index (firs column
                    # of input data)
                    key = major_map[int(parts[self._major_ind])]
                    value = minor_map[int(parts[self._minor_ind])]
                    rating = float(parts[self._r_id])

                    # Populate data dictionary. For each key (major index)
                    # there is a list of value-rating tuples. Example: if the
                    # key is the user, self.data stores for each user, a list
                    # of item-rating pairs.
                    if key not in self.data:
                        self.data[key] = []
                    self.data[key].append((value, rating))
        # print("There has been {} errors when parsing the data".format(error))

    def _build_maps(self):
        """
        Build a mapping between the original input user and item values and an
        internal representation that is used in the model. It generates
        self._user_id_map and self._item_id_map.
        """
        self._user_id_map = dict()
        self._item_id_map = dict()

        # Input file
        src_files = [self._src_file]

        # Loop to read the data and create the internal mapping
        u_id = 0
        i_id = 0
        for source_file in src_files:
            with open(source_file, 'r') as src:
                head = False
                for line in src.readlines():
                    if self._params['header'] and not head:
                        head = True
                        continue
                    parts = line.strip().split(self._delimiter)
                    if len(parts) < 3:
                        raise ValueError(
                            'Encountered badly formatted line in {}'
                            .format(source_file))

                    # Mapping between input user data and internal user
                    # representation
                    u_id_orig = int(parts[self._u_id])
                    if u_id_orig not in self._user_id_map:
                        self._user_id_map[u_id_orig] = u_id
                        u_id += 1

                    # Mapping between input item data and internal item
                    # representation
                    i_id_orig = int(parts[self._i_id])
                    if i_id_orig not in self._item_id_map:
                        self._item_id_map[i_id_orig] = i_id
                        i_id += 1

    def iterate_one_epoch(self, shuffle_data=True):
        """
        Iterate one epoch and yield a minibatch of data.
        The minibatch is a sparse tensor of size (self._batch_size,
        self._vector_dim), if the major index (first column in input data) is
        user, the size would be (minibatch of users, number_items). Most
        entries are zero, because the user has not rated most of the items.
        The non-zero elements are the items rated by the user, the index of
        the element is the item index and the value of the element is the
        rating.
        :return: Minibatch of data as a sparse tensor
        """
        data = self.data
        keys = list(data.keys())
        # We shuffle in the training phase
        if shuffle_data:
            shuffle(keys)

        # Start and end index of the minibatch
        s_ind = 0
        e_ind = self._batch_size

        # Loop to construct the sparse tensor:
        # http://pytorch.org/docs/master/sparse.html#torch-sparse
        # Pytorch sparse tensors are represented in coordinate format. They
        # consist of 2 tensors, a 2D tensor of indices and a tensor of values
        while e_ind < len(keys):
            local_ind = 0  # track the local minibatch index
            inds1 = []  # contain the indices of the minibatch
            # contain the minor index of the input data (second column of the
            # input data)
            inds2 = []
            vals = []  # contain the ratings
            for ind in range(s_ind, e_ind):
                inds2 += [v[0] for v in data[keys[ind]]]
                inds1 += [local_ind]*len([v[0] for v in data[keys[ind]]])
                vals += [v[1] for v in data[keys[ind]]]
                local_ind += 1

            # Set the minibatch as a sparse pytorch tensor, using a 2D tensor
            # of indices i_torch and a tensor of values v_torch
            i_torch = torch.LongTensor([inds1, inds2])
            v_torch = torch.FloatTensor(vals)
            mini_batch = torch.sparse.FloatTensor(
                i_torch, v_torch, torch.Size([self._batch_size,
                                              self._vector_dim]))

            # Yield minibatch
            if shuffle_data:
                yield mini_batch
            else:
                yield mini_batch, keys[s_ind:e_ind]

            # Update loop indexes with batch size
            s_ind += self._batch_size
            e_ind += self._batch_size

    # TODO: refactor, DRY with iterate_one_epoch
    def iterate_one_epoch_eval(self, for_inf=False):
        """
        Iterate one epoch of evaluation data and yield a minibatch.
        The minibatch is composed by two sparse tensors with a similar
        structure as in iterate_one_epoch. The first vector data is the
        evaluation data corresponding to the user-item pairs in the evaluation
        set with the internal representation. The second vector src_data is a
        copy of the user-item pairs in the training set using the internal
        representation. These two sets are used to obtain the loss or the test
        metrics. The use of src_data is needed for the autoencoder to identify
        the user profile, then the autoencoder use this profile to generate the
        ratings of all items for that user. Then the  evaluation data is used
        to compute the loss or test metrics. In this case, the batch_size is
        reduced to 1, in words of the author: to make sure no examples are
        missed.
        :param for_inf: If True returns the minibatch and the major index
                        value (first column in the input data) corresponding
                        to the current minibatch. If False, it only returns
                        the minibatch.
        :return: Minibatch of data as a sparse tensor
        """
        keys = list(self.data.keys())
        s_ind = 0
        while s_ind < len(keys):
            # Indices and value of the evaluation data like in
            # iterate_one_epoch. In this case inds1 is always 0
            # because the minibatch size is 1.
            inds1 = [0] * len([v[0] for v in self.data[keys[s_ind]]])
            inds2 = [v[0] for v in self.data[keys[s_ind]]]
            vals = [v[1] for v in self.data[keys[s_ind]]]

            # Indices and values of the source data
            src_inds1 = [0] * len([v[0] for v in self.src_data[keys[s_ind]]])
            src_inds2 = [v[0] for v in self.src_data[keys[s_ind]]]
            src_vals = [v[1] for v in self.src_data[keys[s_ind]]]

            # Set the sparse pytorch tensor of the evaluation data, using a 2D
            # tensor of indices i_torch and a tensor of values v_torch
            i_torch = torch.LongTensor([inds1, inds2])
            v_torch = torch.FloatTensor(vals)

            # Set the sparse tensor of the source data, which defines the user
            # profile
            src_i_torch = torch.LongTensor([src_inds1, src_inds2])
            src_v_torch = torch.FloatTensor(src_vals)

            mini_batch = (torch.sparse.FloatTensor(i_torch,
                                                   v_torch,
                                                   torch.Size([1,
                                                               self._vector_dim])),
                          torch.sparse.FloatTensor(src_i_torch,
                                                   src_v_torch,
                                                   torch.Size([1,
                                                               self._vector_dim])))
            s_ind += 1
            if not for_inf:
                yield mini_batch
            else:
                yield mini_batch, keys[s_ind - 1]

    @property
    def vector_dim(self):
        """Size of the minor index (the second column of the input data)"""
        return self._vector_dim

    @property
    def user_id_map(self):
        """Mapping between input user data and internal user representation."""
        return self._user_id_map

    @property
    def item_id_map(self):
        """Mapping between input item data and internal item representation."""
        return self._item_id_map

    @property
    def params(self):
        """Parameters of the recommender."""
        return self._params

    @property
    def batch_size(self):
        """Batch size getter"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Batch size setter"""
        self._batch_size = batch_size
