# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import numpy as np
import numba
from sklearn import metrics
from recommenders.models.unirec.facility.evaluation.evaluator_abc import *
from recommenders.models.unirec.facility.evaluation.onepos import OnePositiveEvaluator


@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights


@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, length + 1)
    return mrr_weights


@numba.jit(nopython=True, parallel=True)
def get_top_item_ids(scores, K):
    ## TODO: how to sort only the topk?
    res = np.zeros((len(scores), K), dtype=np.int64)
    for i in numba.prange(len(scores)):
        res[i, :] = np.argsort(scores[i])[::-1][:K]
    return res


def get_top_item_ids_torch(scores, K, device):
    S = torch.from_numpy(scores).to(device)
    res = S.topk(K).indices.cpu().numpy()
    return res


r"""
    MultiPositiveEvaluator only supports OneVSAll evaluation protocol.
"""


class MultiPositiveEvaluator(OnePositiveEvaluator):
    def __init__(self, metrics_str=None, group_size=-1, config=None, accelerator=None):
        print("##--MultiPositiveEvaluator")
        super(MultiPositiveEvaluator, self).__init__(metrics_str, group_size, config, accelerator)
        self.idea_ndcg = {}
        self.max_K = self.get_max_k_value()
        self.ndcg_w = _get_ndcg_weights(self.max_K)
        self.mrr_w = _get_mrr_weights(self.max_K)

    def mrr(self, k, top_ids, pos_itemids, w):
        ## TODO: how to make it parallel?
        N = len(pos_itemids)
        res = np.zeros(N, dtype=np.float32)

        if k not in self.k2overlap_cache:
            self.k2overlap_cache[k] = np.empty(N, dtype=object)

        for i in range(N):
            if self.k2overlap_cache[k][i] is None:
                _, idx01, idx02 = np.intersect1d(
                    pos_itemids[i],
                    top_ids[i][:k],
                    assume_unique=True,
                    return_indices=True,
                )
                self.k2overlap_cache[k][i] = idx02
            else:
                idx02 = self.k2overlap_cache[k][i]
            if len(idx02) > 0:
                res[i] = w[idx02].sum() / min(k, len(pos_itemids[i]))
            else:
                res[i] = 0
        return res

    def get_idea_ndcg(self, k):
        if k in self.idea_ndcg:
            return self.idea_ndcg[k]
        self.idea_ndcg[k] = self.ndcg_w[:k].sum()
        return self.idea_ndcg[k]

    def ndcg(self, k, top_ids, pos_itemids, w):
        ## TODO: how to make it parallel?
        N = len(pos_itemids)
        res = np.zeros(N, dtype=np.float32)

        if k not in self.k2overlap_cache:
            self.k2overlap_cache[k] = np.empty(N, dtype=object)

        for i in range(N):
            if self.k2overlap_cache[k][i] is None:
                _, idx01, idx02 = np.intersect1d(
                    pos_itemids[i],
                    top_ids[i][:k],
                    assume_unique=True,
                    return_indices=True,
                )
                self.k2overlap_cache[k][i] = idx02
            else:
                idx02 = self.k2overlap_cache[k][i]
            if len(idx02) > 0:
                res[i] = w[idx02].sum() / self.get_idea_ndcg(min(k, len(pos_itemids[i])))
            else:
                res[i] = 0
        return res

    def rndcg(self, k, top_ids, pos_itemids, w, item2price):
        ## TODO: how to make it parallel?
        N = len(pos_itemids)
        res = np.zeros(N, dtype=np.float32)

        if k not in self.k2overlap_cache:
            self.k2overlap_cache[k] = np.empty(N, dtype=object)

        for i in range(N):
            if self.k2overlap_cache[k][i] is None:
                _, idx01, idx02 = np.intersect1d(
                    pos_itemids[i],
                    top_ids[i][:k],
                    assume_unique=True,
                    return_indices=True,
                )
                self.k2overlap_cache[k][i] = idx02
            else:
                idx02 = self.k2overlap_cache[k][i]
            for j in idx02:
                _itemid = top_ids[i][j]
                res[i] += w[j] * item2price[_itemid]

            prices = [item2price[t] for t in pos_itemids[i]]
            prices.sort(reverse=True)

            s = 0
            for j in range(min(k, len(pos_itemids[i]))):
                s += w[j] * prices[j]
            res[i] /= s
        return res

    def hit(self, k, top_ids, pos_itemids, item2price=None):
        ## TODO: how to make it parallel?
        N = len(pos_itemids)
        res = np.zeros(N, dtype=np.float32)

        if k not in self.k2overlap_cache:
            self.k2overlap_cache[k] = np.empty(N, dtype=object)

        for i in range(N):
            if self.k2overlap_cache[k][i] is None:
                _, idx01, idx02 = np.intersect1d(
                    pos_itemids[i],
                    top_ids[i][:k],
                    assume_unique=True,
                    return_indices=True,
                )
                self.k2overlap_cache[k][i] = idx02
            else:
                idx02 = self.k2overlap_cache[k][i]

            if len(idx02) > 0:
                if item2price is None:
                    res[i] = 1
                else:
                    _itemid = top_ids[i][idx02]
                    res[i] = item2price[_itemid].max()

        return res

    def recall(self, k, top_ids, pos_itemids, item2price=None):
        ## TODO: how to make it parallel?
        N = len(pos_itemids)
        res = np.zeros(N, dtype=np.float32)

        if k not in self.k2overlap_cache:
            self.k2overlap_cache[k] = np.empty(N, dtype=object)

        for i in range(N):
            if self.k2overlap_cache[k][i] is None:
                _, idx01, idx02 = np.intersect1d(
                    pos_itemids[i],
                    top_ids[i][:k],
                    assume_unique=True,
                    return_indices=True,
                )
                self.k2overlap_cache[k][i] = idx02
            else:
                idx02 = self.k2overlap_cache[k][i]

            n_hit = len(idx02)
            if n_hit > 0:
                if item2price is None:
                    res[i] = n_hit / len(pos_itemids[i])
                else:
                    _itemid = top_ids[i][idx02]
                    res[i] = item2price[_itemid].sum()

        return res

    def get_max_k_value(self):
        max_k = 0
        for metric in self.metrics_list:
            if "@" in metric:
                tokens = metric.split("@")
                key, ks = tokens[0], tokens[1].split(";")
                for k in ks:
                    max_k = max(int(k), max_k)
        return max_k

    def convert_2_set(self, pos_itemids):
        res = np.empty(len(pos_itemids), dtype=object)
        for i in range(len(pos_itemids)):
            res[i] = set(pos_itemids[i])
        return res

    ## TODO: computing AUC is very slow
    def compute_AUC(self, S, pos_itemids):
        n, m = S.shape
        res = np.zeros(n, dtype=np.float32)
        for i in range(n):
            labels = np.zeros(m, dtype=np.int32)
            labels[pos_itemids[i]] = 1
            res[i] = metrics.roc_auc_score(labels, S[i])
        return res

    def evaluate_with_scores(self, scores, labels=None, pos_itemids=None, **kwargs):
        S = scores
        res = {}
        max_K = self.max_K

        # remove padding items to ensure unique ground-truth
        pos_itemids = self.remove_padding_items(pos_itemids)

        # add small perturbation
        shape_key = S.shape
        if shape_key not in self.noise:
            self.noise[shape_key] = np.random.uniform(low=-1e-8, high=1e-8, size=S.shape)
        S += self.noise[shape_key]

        top_ids = get_top_item_ids_torch(S, max_K, self.config.get("device", "cpu"))

        ## Cache the overlap items for speedup. We need to clean the cache each time.
        self.k2overlap_cache = {}
        for metric in self.metrics_list:
            if "group_auc" == metric:
                res[metric] = self.compute_AUC(S, pos_itemids)
            # elif 'ndcg' == metric:
            #     res['ndcg'] = self.ndcg(np.Inf, rank, ndcg_w)
            # elif 'mrr' == metric:
            #     res['mrr'] = mrr_w[rank]
            elif "@" in metric:
                tokens = metric.split("@")
                key, ks = tokens[0], tokens[1].split(";")
                if key == "ndcg":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.ndcg(int(k), top_ids, pos_itemids, self.ndcg_w)
                elif key == "rndcg":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.rndcg(int(k), top_ids, pos_itemids, self.ndcg_w, self.item2price)
                elif key == "hit":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.hit(int(k), top_ids, pos_itemids)
                elif key == "rhit":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.hit(int(k), top_ids, pos_itemids, self.item2price)
                elif key == "recall":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.recall(int(k), top_ids, pos_itemids)
                elif key == "rrecall":
                    for k in ks:
                        res["{0}@{1}".format(key, k)] = self.recall(int(k), top_ids, pos_itemids, self.item2price)
                elif key == "mrr":
                    for k in ks:
                        k = int(k)
                        res["{0}@{1}".format(key, k)] = self.mrr(int(k), top_ids, pos_itemids, self.mrr_w)
                else:
                    raise ValueError("metric {0} is unknown.".format(key))

        return res
