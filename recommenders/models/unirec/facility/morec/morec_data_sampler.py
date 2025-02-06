# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import torch
import logging
import numpy as np
import pandas as pd
from typing import *
from accelerate import Accelerator
import torch.nn as nn
from torch.utils.data import Sampler, BatchSampler, Dataset, DataLoader

from recommenders.models.unirec.utils.general import pad_sequence_arrays


def normalize(data):
    return data / (data.sum() + 1e-10)


class BaseSampler(BatchSampler):
    """
    BaseSampler is an iterator which could be used as a `batch_sampler` in DataLoader.
    It iterates a batch of sample index each time. And BaseSampler could only handle uniformly
    sampling. It works with `num_workers` in DataLoader because each worker aims to load
    a batch of samples each time.
    """

    def __init__(self, config: Dict, train_data, shuffle: bool = True) -> None:
        # In order to be compatible to new-version accelerate, we need to inherit from PyTorch's BatchSampler.
        # Actually it's not required in original PyTorch. More details refer to the issue https://github.com/huggingface/accelerate/issues/2091.
        super().__init__(Sampler(train_data), config["batch_size"], False)
        self.config = config
        self.train_data = train_data
        self.model = None
        self.accelerator = None
        self.batch_size = self.config["batch_size"]
        self.shuffle = shuffle
        self.logger = logging.getLogger(config["exp_name"])
        self.data_index = np.arange(len(self.train_data))  # train data index
        self._post_process()

    def set_model(self, model: nn.Module, accelerator: Accelerator = None):
        self.model = model
        self.accelerator = accelerator

    def set_data_index(self, data_index: np.ndarray):
        self.data_index = data_index

    def _post_process(self):
        pass

    def __iter__(self):
        batch_num = len(self)
        if self.shuffle:
            index = np.random.permutation(self.data_index)
        else:
            # np.random.rand(1) # to keep alignment for random state
            index = self.data_index

        output = np.array_split(index, batch_num)
        yield from output

    def __len__(self):
        return (len(self.data_index) + self.batch_size - 1) // self.batch_size

    def optimize(self):
        pass

    def get_loader(self, num_workers=4):
        loader = DataLoader(self.train_data, batch_sampler=self, num_workers=num_workers)
        if self.accelerator is not None:
            return self.accelerator.prepare(loader)
        else:
            return loader


class MoRecDS(BaseSampler):
    def __init__(
        self,
        config: Dict,
        objectives: list,
        ngroup: int,
        train_data: Dataset,
        val_data: Dataset,
        alpha: Union[Dict, float],
        item2meta: pd.DataFrame = None,
        align_dist: np.ndarray = None,
        # item2category: np.ndarray=None,
        # item2pop: np.ndarray=None,
        user2history: np.ndarray = None,
        topk: int = 100,
        fairness_metric: str = "loss",
    ) -> None:
        """
        MoRec data sampler proposed in our paper "" for unified objective modeling.

        Args:
            config: the global configuration dict
            objectives (list): objectives to be optimized, available objectives: [revenue, fairness, alignment]
            ngroup (list or int): samples are divided into several groups, aiming to provide efficiency (revenue, alignment).
                                  If it's a positive integer, number of groups for revenue and alignment would be the same.
                                  If it's a non-positive integer, number of groups for alignment would be set as 10 by default,
                                  and samples would not be divided into groups for revenue.
                                  If it's a list, the i-th ngroup represents the number of group of the i-th objective in `objectives`.
            train_data (Dataset): training set, used for building index in each group.
            val_data (Dataset): validation set, used for obtain signals of the update of sampling weights
            alpha (float): learning rate of signed SGD for updating sampling weights
            item2price (numpy.ndarray): the price of item, used for setting the sampling weight for revenue.
            item2category (numpy.ndarray): the category of item, used for updating the sampling weight for fairness
            item2pop (numpy.ndarray): the popularity of item, used as the preset distribution for alignment
            user2history (numpy.ndarray): the history of user, used in topk operation (topk is used to get recommendation for alignment)
            topk (int): the topk number for obtain recommendation list for alignment
            fairness_metric (str): the metric used to monitor the least misery in fairness, optional values: [loss, hit]
        """

        super().__init__(config, train_data, True)
        self.config = config
        self.objectives = objectives
        self.val_data = val_data
        self.item2group = {}
        self.group2info = {}
        self.topk = topk
        self.item2meta = item2meta
        self.user2history = user2history
        self.alpha = alpha if isinstance(alpha, dict) else {ob: alpha for ob in objectives}
        self.fairness_metric = fairness_metric

        # we leave 0 as the padding index of group, so ngroup is set to group_id.max() + 1 for each objective
        # More details are refered to the `load_morec_meta_data` funtion in `unirec/facility/morec/__init__.py`
        self.ngroup = {}
        for obj in objectives:
            if obj in {"fairness", "alignment"}:
                col = "fair_group" if obj == "fairness" else "align_group"
                self.item2group[obj] = self.item2meta.loc[np.arange(self.config["n_items"])][col].to_numpy()
                self.ngroup[obj] = self.item2group[obj].max() + 1
            elif obj in {"revenue"}:
                item2weight = self.item2meta.loc[np.arange(self.config["n_items"])]["weight"].to_numpy()
                if ngroup > 0:
                    _item2group, _group2info = self.group_item_by_attr(item2weight, ngroup, zero_as_group=False)
                else:
                    _item2group = np.arange(self.config["n_items"])
                    _group2info = item2weight
                self.item2group["revenue"] = _item2group
                self.group2info["revenue"] = _group2info
                self.ngroup["revenue"] = _item2group.max() + 1  # including padding group index
            else:
                raise ValueError(f"Only support ['fairness', 'alignment', 'revenue'] objectives, while got {obj}.")

        self.align_target_dist = np.concatenate(
            (
                np.zeros(
                    1,
                ),
                align_dist,
            ),
            axis=0,
        )

        self.group2dataindex_trn = {}
        self.group2dataindex_val = {}
        self.group2weights = {}
        for ob in objectives:
            _item2group = self.item2group[ob]
            self.group2dataindex_trn[ob], self.group2weights[ob] = self._get_group_data_index(train_data, _item2group)
            self.group2dataindex_val[ob], _ = self._get_group_data_index(val_data, _item2group)

        # initialize sampling weight for revenue
        self.group2weights["revenue"] = normalize(self.group2info["revenue"])

    @staticmethod
    def group_item_by_attr(item2info: np.ndarray, ngroup: int, zero_as_group: bool = True):
        r"""Group items according to item's information

        Args:
            item2info(np.ndarray): information of item, such as price, popularity. The shape
                is (#items,)
            ngroup(int): number of groups
            zero_as_group(bool): whether items with zero values be grouped into one group.

        Return:
            item2group(np.ndarray): group id of each item
            group2info(np.ndarray): the average information value of group
        """
        if zero_as_group:
            zero_idx = np.squeeze(np.argwhere(item2info == 0.0), axis=-1)
            _ngroup = ngroup - 1
            valid_len = len(item2info) - len(zero_idx)
        else:
            _ngroup = ngroup
            valid_len = len(item2info)
        idx = np.argsort(-item2info, -1, "stable")
        groups = np.array_split(idx[:valid_len], _ngroup)
        if zero_as_group:
            groups.append(zero_idx)

        # assign group id to each item
        g_id = np.arange(valid_len)
        _split_index = (valid_len // ngroup + 1) * (valid_len % ngroup)
        g_id[g_id < _split_index] //= valid_len // ngroup + 1
        g_id[_split_index:] = (g_id[_split_index:] - _split_index) // (valid_len // ngroup) + (valid_len % ngroup)
        g_id += 1  # g_id start from 1, leave 0 as padding index

        g_id[1 : valid_len % ngroup + 1] += 1  # group_id from 1, 0 saved as padding
        item2gid = np.zeros_like(item2info, dtype=int)
        item2gid[idx[:valid_len]] = g_id
        if zero_as_group:
            item2gid[zero_idx] = _ngroup + 1

        group2info = np.zeros(ngroup + 1)
        for i in range(1, ngroup + 1):
            _idx = item2gid == i
            if _idx.sum() > 0:
                group2info[i] = item2info[item2gid == i].mean()
        return item2gid, group2info

    def _get_group_data_index(self, data, item2group: np.ndarray):
        r"""Get data index in data for each group.

        Args:
            data(Dataset): train data or valid data
            item2group(np.ndarray): group of each item

        Return:
            group2index(list): index of each group. e.g. the i-th element is the data index of i-th group
        """
        item_col = data.dataset[:, data.return_key_2_index["item_id"]]
        group_col = item2group[item_col.astype(int)]
        ngroup = int(max(item2group) + 1)
        group2index = [None] * ngroup
        group2ratio = np.zeros(ngroup)
        for i in range(1, ngroup):  # only update groups with index from 1 to group_id.max(), ignore padding index 0
            group2index[i] = np.squeeze(np.argwhere(group_col == i), axis=1)
            group2ratio[i] = len(group2index[i]) / len(item_col) if not (group2index[i] is None) else 0
        return group2index, group2ratio

    def _cal_fair_signal(self, topk_items=None, target_items=None) -> np.ndarray:
        r"""calculate the update of sampling weight for fairness with loss or hit metric."""
        if (topk_items is None) or (target_items is None):
            group2index = self.group2dataindex_val["fairness"]
            base_sampler = BaseSampler(self.config, self.val_data, shuffle=False)
            base_sampler.set_model(self.model, self.accelerator)
            ngroup = self.ngroup["fairness"]
            loss = np.zeros(ngroup)
            loss[0] = -np.inf
            for groupid in range(1, ngroup):  # only update groups with index from 1 to group_id.max(), ignore padding index 0
                group_data_index = group2index[groupid]
                if len(group_data_index) <= 0:
                    loss[groupid] = -np.inf
                else:
                    base_sampler.set_data_index(group_data_index)
                    group_data_loader = base_sampler.get_loader(4)
                    loss[groupid] = self._gather_loss(self.model, group_data_loader)  # a scaler

            signal = np.zeros(ngroup)
            max_loss_group_id = np.argmax(loss)
            signal[max_loss_group_id] = 1
        else:  # use hit metric as the fairness judgement
            ngroup = self.ngroup["fairness"]
            item2group = self.item2group["fairness"]
            hit = np.any(topk_items[:, :10] == target_items[:, None], axis=-1)
            group_id = item2group[target_items]
            group2hit = np.zeros(ngroup)
            for i in range(1, ngroup):  # only update groups with index from 1 to group_id.max(), ignore padding index 0
                _idx = group_id == i
                if _idx.sum() > 0:
                    group2hit[i] = hit[_idx].sum() / _idx.sum()
            group2hit[0] = 1.0  # mask padding idx
            signal = np.zeros(ngroup)
            min_hit_group_id = np.argmin(group2hit)
            signal[min_hit_group_id] = 1
        return signal

    @torch.no_grad()
    def _gather_loss(self, model, data):
        """gather mean loss for the given data"""
        total_loss = 0.0
        total_bs = 0
        model.train()
        for idx, inter_data in enumerate(data):
            samples = {k: inter_data[v] for k, v in data.dataset.return_key_2_index.items()}
            batch_size = inter_data[0].shape[0]
            loss, scores, _, _ = self.model(**samples)
            # loss = self.model._cal_loss(scores, samples['label'])
            loss_sum = batch_size * loss.data
            total_loss += self.accelerator.gather_for_metrics(loss_sum).sum()
            total_bs += batch_size
        return total_loss / len(data.dataset)

    def _cal_pop_bias_signal(self, topk_items: np.ndarray) -> np.ndarray:
        r"""calculate signal for popularity debias"""
        all_topk_items = topk_items.reshape(-1)
        item_id, counts = np.unique(all_topk_items, return_counts=True)

        item2group = self.item2group["alignment"]
        group_id = item2group[item_id]

        ngroup = self.ngroup["alignment"]
        group2counts = np.zeros(self.ngroup["alignment"])
        for i in range(ngroup):
            _idx = group_id == i
            if _idx.sum() > 0:  # padding group index 0 is ignored here
                group2counts[i] = counts[_idx].sum()
        group2pop = group2counts / group2counts.sum()

        signal = np.zeros(
            ngroup,
        )
        div = group2pop - self.align_target_dist
        signal[div > 0] = -1
        signal[div < 0] = 1
        return signal

    @torch.no_grad()
    def _gather_topk(self, model, data, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        all_topk_items = []
        target_items = []
        for idx, inter_data in enumerate(data):
            samples = {k: inter_data[v] for k, v in data.dataset.return_key_2_index.items()}
            pos_item_id = samples["item_id"][:, 0]
            user_id = samples["user_id"].cpu()
            user_hist = self.user2history[user_id]
            user_hist = torch.from_numpy(pad_sequence_arrays(user_hist).astype(int))
            user_hist[user_hist == pos_item_id.cpu().unsqueeze(-1)] = 0  # Do not mask items in valid data in topk operation
            # samples = model.collect_data(inter_data, schema=data.dataset.return_key_2_index)
            topk_scores, topk_items = model.topk(samples, k, user_hist)
            pos_item_id = self.accelerator.gather_for_metrics(pos_item_id)
            topk_items = self.accelerator.gather_for_metrics(topk_items)
            target_items.append(pos_item_id.cpu())
            all_topk_items.append(topk_items.cpu())
        all_topk_items = torch.cat(all_topk_items, dim=0)  # [#interactions, k]
        target_items = torch.cat(target_items, dim=0)
        return all_topk_items.numpy(), target_items.numpy()

    def gather_signals(self) -> np.ndarray:
        r"""gather validation signals for each objective"""
        base_sampler = BaseSampler(self.config, self.val_data, shuffle=False)
        base_sampler.set_model(self.model, self.accelerator)
        val_loader = base_sampler.get_loader(4)
        topk_items, target_items = self._gather_topk(self.model, val_loader, k=self.topk)

        signal = {}
        # fairness update signal
        if "fairness" in self.objectives:
            if self.fairness_metric == "hit":
                signal["fairness"] = self._cal_fair_signal(topk_items, target_items)
            else:
                signal["fairness"] = self._cal_fair_signal()
        else:
            # fair_signal = np.zeros(self.ngroup['fairness'])
            signal["fairness"] = None

        # revenue signal
        if "revenue" in self.objectives:
            signal["revenue"] = np.zeros(self.ngroup["revenue"])
        else:
            signal["revenue"] = None

        # debias update signal
        if "alignment" in self.objectives:
            signal["alignment"] = self._cal_pop_bias_signal(topk_items)
        else:
            signal["alignment"] = None

        return signal

    def optimize(self):
        r"""adjust weight for each objectives.
        The method should be called each epoch. Use signed-SGD.
        """
        signals = self.gather_signals()  # loss for each group col

        for ob in self.objectives:
            alpha = self.alpha[ob]
            signal = signals[ob]

            descending_g_id = np.argwhere(signal < 0)
            ascending_g_id = np.argwhere(signal > 0)
            #
            # bound all weights into [0,1]
            des_num, asc_num = len(descending_g_id), len(ascending_g_id)
            if des_num > 0 and asc_num > 0:
                delta = alpha
                self.group2weights[ob][descending_g_id] -= delta
                self.group2weights[ob][ascending_g_id] += delta
                self.group2weights[ob][self.group2weights[ob] <= 0] = 0.0
            elif des_num > 0 and asc_num == 0:
                delta = min([alpha, self.group2weights[ob][descending_g_id]])
                self.group2weights[ob][descending_g_id] -= delta
            elif des_num == 0 and asc_num > 0:
                delta = alpha
                self.group2weights[ob][ascending_g_id] += delta
            elif des_num == 0 and asc_num == 0:  # add a random noise for the unchanged weight
                pass
            self.group2weights[ob] = normalize(self.group2weights[ob])
        return

    def __iter__(self):
        if self.model is None:
            self.logger.warning("The `model` is not set for MoRecDS.")
        else:
            self.optimize()
        groupcol2data_index = []
        for ob in self.objectives:
            group_batch_size = np.floor(self.group2weights[ob] * self.batch_size).astype(int)
            group_batch_size[-1] = self.batch_size - (group_batch_size[:-1].sum())
            data_index = []
            select_g_index = [None] * (len(self.group2dataindex_trn[ob]) - 1)
            for i, g_index in enumerate(self.group2dataindex_trn[ob][1:]):
                if len(g_index) > 0:
                    select_g_index[i] = self._sample_batch(group_batch_size[i + 1], g_index, len(self), True)
                else:  # no sample corresponding to this group
                    select_g_index[i] = self._sample_batch(
                        group_batch_size[i + 1],
                        np.arange(len(self.train_data)),
                        len(self),
                        False,
                    )
            data_index = np.concatenate(select_g_index, axis=-1)  # batch_num * batch_size

            data_index = np.random.permutation(data_index)
            groupcol2data_index.append(data_index)

        random_data_index = self._sample_batch(self.batch_size, np.arange(len(self.train_data)), len(self), False)
        groupcol2data_index.append(random_data_index)
        data_index = self._stack_col_index(groupcol2index=groupcol2data_index)
        yield from data_index

    def _sample_batch(self, batch_size, data_index, batch_num, replace=True):
        r"""sample batches with given data_index."""
        if replace:
            select_index = np.random.choice(data_index, size=(batch_num, batch_size), replace=True)
        else:
            select_index = []
            shuffeled_data_index = np.random.permutation(data_index)
            start_idx = 0
            for i in range(batch_num):
                if start_idx + batch_size > len(data_index):
                    _idx = np.concatenate(
                        (
                            shuffeled_data_index[start_idx:],
                            shuffeled_data_index[: batch_size - (len(data_index) - start_idx)],
                        )
                    )
                    select_index.append(_idx)
                    start_idx = batch_size - (len(data_index) - start_idx)
                else:
                    select_index.append(shuffeled_data_index[start_idx : start_idx + batch_size])
                    start_idx += batch_size
            select_index = np.stack(select_index, axis=0)

        return select_index

    def _stack_col_index(self, groupcol2index):
        stack_index = [None] * len(self)
        for i in range(len(self)):
            _idx = []
            for index in groupcol2index:
                if len(index) > 0:
                    _idx.append(index[i])
            stack_index[i] = np.concatenate(_idx)
        return np.stack(stack_index, axis=0)


if __name__ == "__main__":
    config = {}
    objectives = ["fairness", "alignment", "revenue"]

    pass
