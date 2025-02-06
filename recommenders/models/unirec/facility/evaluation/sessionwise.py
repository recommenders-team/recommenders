# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import numpy as np
import numba
from collections import defaultdict
from sklearn import metrics
from tqdm import tqdm

from recommenders.models.unirec.facility.evaluation.evaluator_abc import *


@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights


@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, length + 1)
    return mrr_weights


@numba.jit(nopython=True, parallel=True)
def get_ranks(scores, labels):
    id2rank = np.argsort(
        scores,
    )[::-1]
    ranks = np.zeros_like(id2rank)
    ranks[id2rank] = np.arange(len(id2rank))
    return ranks[labels > 0]


class SessionWiseEvaluator(Evaluator):
    def __init__(
        self,
        metrics_str=None,
        group_size=-1,
        config=None,
        total_session=0,
        accelerator=None,
    ):
        super(SessionWiseEvaluator, self).__init__(metrics_str, group_size, config, accelerator)
        self.total_session = total_session
        # self.metrics_list = eval(metrics_str)
        self.group_size = group_size

    r"""
       Items can have prices. If price is not None, compute the G-NDCG@k (GMV variants of NDCG) metrics as definied in this paper:
        'A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation.'
    """

    def ndcg(self, k, ranks, w, price=None):
        n = min(k, len(ranks))
        masker = ranks < k
        if price is None:
            res = np.sum(w[ranks[masker]]) / np.sum(w[:n])
        else:
            res = np.sum(w[ranks[masker]] * price[masker]) / (np.sum(w[:n] * np.sort(price)[::-1][:n]) + 1e-8)  # in case of price=0
        return res

    def mrr(self, k, ranks, w):
        n = min(k, len(ranks))
        masker = ranks < k
        res = np.sum(w[ranks[masker]]) / n
        return res

    r"""
       Items can have prices. If price is not None, compute the G-MAP@k (with slightly modifications here: return the max price item in hit) metrics as definied in this paper:
        'A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation.'
    """

    def rhit(self, k, ranks, prices):
        v = ranks < k
        return max((v + 0) * prices)

    def hit(self, k, ranks):
        if ranks[0] < k:
            return 1.0
        else:
            return 0.0

    def recall(self, k, ranks):
        if len(ranks) <= 0:
            return 0
        return len(ranks[ranks < k]) / len(ranks)

    r"""
       Items can have prices. If price is not None, compute the G-MAP@k (with slightly modifications here) metrics as definied in this paper:
        'A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation.'
    """

    def rrecall(self, k, ranks, prices):
        v = ranks < k
        return sum((v + 0) * prices)

    def roc_auc(self, g_scores, g_labels):
        # try:
        #     return metrics.roc_auc_score(g_labels, g_scores)
        # except:
        #     raise ValueError('GT label values incorrect: {0}'.format(g_labels))
        return metrics.roc_auc_score(g_labels, g_scores)

    def group_scores_by_session(self, scores, labels, session_ids, prices=None):
        res_scores, res_labels, res_prices = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        if prices is None:
            for score, label, session in zip(scores, labels, session_ids):
                res_scores[session].append(score)
                res_labels[session].append(label)
        else:
            for score, label, session, price in zip(scores, labels, session_ids, prices):
                res_scores[session].append(score)
                res_labels[session].append(label)
                res_prices[session].append(price)
        print("[+] Original session count: {0}".format(len(res_scores)))

        ## remove sessions that is all positive or all negative
        valid_scores, valid_labels, valid_prices = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list) if prices is not None else None,
        )
        for k, v in res_labels.items():
            n_pos = sum(v)
            if n_pos <= 0 or n_pos == len(v):
                continue
            valid_labels[k] = np.array(v)
            valid_scores[k] = np.array(res_scores[k])
            if prices is not None:
                valid_prices[k] = np.array(res_prices[k])

        print("[+] Filtered session count: {0}".format(len(valid_scores)))
        return valid_scores, valid_labels, valid_prices

    def _get_ndcg_weights(self, n):
        if not hasattr(self, "ndcg_w") or len(self.ndcg_w) < n:
            self.ndcg_w = _get_ndcg_weights(n)
        return self.ndcg_w[:n]

    def _get_mrr_weights(self, n):
        if not hasattr(self, "mrr_w") or len(self.mrr_w) < n:
            self.mrr_w = _get_mrr_weights(n)
        return self.mrr_w[:n]

    def evaluate_with_scores(self, scores, labels=None, session_ids=None, prices=None, **kwargs):
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        if prices is not None:
            prices = prices.reshape(-1)

        # add small perturbation
        noise = np.random.uniform(low=-1e-8, high=1e-8, size=(1000,))
        n_repeat = len(scores) // len(noise) + 1
        noise = np.broadcast_to(noise, shape=(n_repeat, len(noise)))
        scores += noise.reshape(-1)[: len(scores)]

        session2scores, session2labels, session2prices = self.group_scores_by_session(scores, labels, session_ids, prices)

        need_ranks = self._need_ranks(self.metrics_list)
        res = defaultdict(list)
        session_idx, n_sessions = 0, len(session2scores)
        iter_data = tqdm(session2scores, desc="Calc by sessions", dynamic_ncols=True) if self.config["verbose"] == 2 else session2scores
        for session_id in iter_data:
            g_scores, g_labels = session2scores[session_id], session2labels[session_id]
            if prices is not None:
                g_prices = session2prices[session_id]
            else:
                g_prices = None

            if need_ranks:
                ranks = get_ranks(g_scores, g_labels)
                ranks_indices = np.argsort(ranks)

                # ranks.sort()
                ranks = ranks[ranks_indices]
                if prices is not None:
                    rank_prices = g_prices[g_labels > 0][ranks_indices]

                n_group_items = len(g_scores)
                ndcg_w = self._get_ndcg_weights(n_group_items)
                mrr_w = self._get_mrr_weights(n_group_items)

            for metric in self.metrics_list:
                if "group_auc" == metric:
                    res[metric].append(self.roc_auc(g_scores, g_labels))
                elif "ndcg" == metric:
                    res["ndcg"].append(self.ndcg(np.Inf, ranks, ndcg_w))
                elif "rndcg" == metric:
                    res["rndcg"].append(self.ndcg(np.Inf, ranks, ndcg_w, rank_prices))
                elif "mrr" == metric:
                    res["mrr"].append(self.mrr(np.Inf, ranks, mrr_w))
                elif "@" in metric:
                    tokens = metric.split("@")
                    key, ks = tokens[0], tokens[1].split(";")
                    if key == "ndcg":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.ndcg(int(k), ranks, ndcg_w))
                    elif key == "rndcg":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.ndcg(int(k), ranks, ndcg_w, rank_prices))
                    elif key == "hit":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.hit(int(k), ranks))
                    elif key == "rhit":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.rhit(int(k), ranks, rank_prices))
                    elif key == "recall":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.recall(int(k), ranks))
                    elif key == "rrecall":
                        for k in ks:
                            res["{0}@{1}".format(key, k)].append(self.rrecall(int(k), ranks, rank_prices))
                    elif key == "mrr":
                        for k in ks:
                            k = int(k)
                            res["{0}@{1}".format(key, k)].append(self.mrr(k, ranks, mrr_w))
                    else:
                        raise ValueError("metric {0} is unknown.".format(key))

                session_idx += 1

        if (
            self.total_session > 0
        ):  # total_session means the number of sessions in the test set, including the sessions that have no positive items
            for key, value in res.items():
                if key.startswith("mrr") or key.startswith("ndcg") or key.startswith("hit"):
                    res[key].extend([0.0] * (self.total_session - n_sessions))
        return res

    def merge_scores(self, res):
        all_res = {}
        for key, value in res.items():
            all_res[key] = sum(value) / len(value)
        return all_res


if __name__ == "__main__":
    config = {"verbose": 2, "data_format": "user-item-label-session"}
    evaluator = SessionWiseEvaluator(
        "['group_auc', 'hit@1;5', 'rhit@1;5', 'ndcg@1;5', 'rndcg@1;5', 'ndcg', 'mrr', 'mrr@1;5']",
        config=config,
    )
    data = np.random.uniform(low=-1e-8, high=1e-8, size=(100, 20))
    prices = np.random.uniform(low=0, high=100, size=(100, 20))
    pos_addon = np.random.uniform(low=0.3, high=0.7, size=(100, 3))
    neg_addon = np.random.uniform(low=0.1, high=0.5, size=(100, 17))
    data[:, :3] += pos_addon
    data[:, 3:] += neg_addon
    # labels = np.random.binomial(1, 0.5, data.shape).reshape(data.shape)
    labels = np.zeros_like(data)
    labels[:, :3] = 1
    session_ids = np.arange(len(data), dtype=np.int32)
    session_ids = np.repeat(session_ids, data.shape[1]).reshape(-1).tolist()
    res = evaluator.evaluate_with_scores(data, labels, session_ids, prices)
    b = evaluator.merge_scores(res)
    print(b)

    from sklearn.metrics import ndcg_score, roc_auc_score

    print("sklearn ndcg: {0}".format(ndcg_score(labels, data)))
    print("sklearn ndcg@1: {0}".format(ndcg_score(labels, data, k=1)))
    print("sklearn ndcg@5: {0}".format(ndcg_score(labels, data, k=5)))

    print("sklearn AUC: {0}".format(roc_auc_score(labels.reshape(-1), data.reshape(-1))))
    print("sklearn AUC: {0}".format(np.mean(np.array([roc_auc_score(label, score) for label, score in zip(labels, data)]))))
