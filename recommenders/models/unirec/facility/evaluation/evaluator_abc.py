# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import inspect
from scipy import sparse as ssp
from ast import literal_eval

from recommenders.models.unirec.constants.protocols import *
from recommenders.models.unirec.constants.loss_funcs import *
from recommenders.models.unirec.constants.global_variables import EPS
from recommenders.models.unirec.utils.file_io import load_pkl_obj
from recommenders.models.unirec.utils import general


## all the supported evaluation metrics
SUPPORTED_RANKING_METRICS = {
    "group_auc",
    "ndcg",
    "rndcg",
    "mrr",
    "hit",
    "rhit",
    "recall",
    "rrecall",
}

## metrics that request the computation of ranks
METRICS_NEED_RANKS = {"ndcg", "rndcg", "mrr", "hit", "rhit", "recall", "rrecall"}

## metrics that are related to revenue
METRICS_NEED_PRICE = {"rndcg", "rhit", "rrecall"}

## metrics that are related to fairness, which request category information
METRICS_NEED_CATE = {"least-misery"}

## metrics that request the topk items id
METRICS_NEED_TOPK = {"pop-kl"}


def tensor2array(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data


class Evaluator(object):
    def __init__(self, metrics_str="[auc]", group_size=-1, config=None, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.NINF = -9999  # np.NINF
        self.sup_metrics = SUPPORTED_RANKING_METRICS
        self.metrics_need_ranks = METRICS_NEED_RANKS
        self.item2meta_morec: pd.DataFrame = None
        self.alignment_dist: np.ndarray = None
        self.metrics_list = literal_eval(metrics_str)

        self.metrics_name = {m.split("@")[0] for m in self.metrics_list}

        if len(self.metrics_name.intersection(METRICS_NEED_TOPK)) > 0:
            for m in literal_eval(metrics_str):
                if m.startswith("pop-kl"):
                    self._max_cutoff = max([int(k) for k in m.split("@")[1].split(";")])
            self._topk_flag = True
        else:
            self._topk_flag = False

    def evaluate_with_scores(self, scores, labels=None, **kwargs):
        raise NotImplementedError

    def _need_ranks(self, metrics_list):
        for metric in metrics_list:
            name = metric.split("@")[0]  ## metric string is like ``name@k``
            if name in self.metrics_need_ranks:
                return True
        return False

    # r"""
    #     Load item2price from file. The file should be csv-type file(with header and sep=',') with only two columns-['item_id', 'price'].
    #     In addition, pickle/feather file which contains dataframe is supported.
    #     The dtype of the 'item_id' column should be int and dtype of 'price' column should be float.
    #     item2price is supposed to be a 1-D ndarray, where item2price[i] means the price for item i.
    # """
    # def set_item2price(self, item_price_file: str) -> None:
    #     self.item2price = general.load_item2info(self.config['n_items'], item_price_file, 'price')

    # r"""
    #     Load item2category from file. The file should be csv-type file(with header and sep=',') with only two columns-['item_id', 'category'].
    #     In addition, pickle/feather file which contains dataframe is supported.
    #     The dtype of the 'item_id' column should be int and dtype of 'category' column should be int.
    #     item2category is supposed to be a 1-D ndarray, where item2category[i] means the category for item i.
    #     For example, the category information may be like 'puzzle' 'shooting' when items are games, while the word should be mapped to ids.

    #     ?? Why we need the information?
    #     >> For fairness evaluation, we aims to compare performance of various groups or improve the worst group, so we need the category
    #        information as the group indentifier. We would support user2category in the futher when necessary.
    # """
    # def set_item2category(self, item_cate_file: str) -> None:
    #     self.item2category = general.load_item2info(self.config['n_items'], item_cate_file, 'category')

    r"""
        Set item2popularity from numpy object. 

        ?? Why we need the information?
        >> In debiasing objective, there is a task that proposed in MoRec aiming to align the distribution of topk items with 
           specific distribution. The distribution is the frequency of items in each group, where groups are obtained by dividing
           items according to their popularities (items with similar popularities are grouped in the same group). 
    """

    def set_item_meta_morec(self, item2meta: pd.DataFrame, align_distribution: np.ndarray) -> None:
        """Set item popularity information.

        The information consists of the frequency of appearance of the item in the data set and the group id of items,
        where the groups are divided by the original popularity. That means items with similarity popularity would be
        divided in the same group.

        Args:
            - item2pop(np.ndarray): popularity of items or popularity group id of items, depending on the `is_group` arguments.
            - item2pop_group(np.ndarray): group id of items.
        """
        # if is_group and (max_group_id > 100):
        #     print(f"Warning: the max group id of popularity is {max_group_id}, please check whether it's group id or the original popularity.")
        self.item2meta_morec = item2meta
        self.alignment_dist = align_distribution

    @torch.no_grad()
    def evaluate(self, data, model, verbose=0, predict_only=False):
        model.eval()
        model = self.accelerator.unwrap_model(model)

        iter_data = (
            tqdm(
                enumerate(data),
                total=len(data),
                desc="Evaluate",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )
            if verbose == 2
            else enumerate(data)
        )

        all_scores = []
        all_labels, label_index = (
            [],
            data.dataset.return_key_2_index[ColNames.LABEL.value],
        )
        if data.dataset.config["data_format"] == DataFileFormat.T2_1.value:
            all_session_ids = []
            session_id_idx = data.dataset.return_key_2_index[ColNames.SESSION.value]
        else:
            all_session_ids = None
        if len(self.metrics_name.intersection(METRICS_NEED_PRICE)):
            all_prices = []
            item_id_idx = data.dataset.return_key_2_index[ColNames.ITEMID.value]
        else:
            all_prices = None

        for _, inter_data in iter_data:
            samples = {k: inter_data[v] for k, v in data.dataset.return_key_2_index.items()}
            scores = model.predict(samples)
            labels = inter_data[label_index]
            if all_session_ids is not None:
                session_ids = inter_data[session_id_idx]
            if all_prices is not None:
                items = inter_data[item_id_idx]

            scores = self.accelerator.gather_for_metrics(torch.tensor(scores, device=self.accelerator.device)).cpu().numpy()
            labels = self.accelerator.gather_for_metrics(labels).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)
            if all_session_ids is not None:
                session_ids = self.accelerator.gather_for_metrics(session_ids).cpu().numpy()
                all_session_ids.extend(session_ids.tolist())
            if all_prices is not None:
                items = self.accelerator.gather_for_metrics(items).cpu().numpy()
                all_prices.append(self.item2meta_morec.loc[items]["weight"].to_numpy())

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        if all_prices is not None:
            all_prices = np.concatenate(all_prices)
        if predict_only:
            return all_scores
        else:
            result = self.evaluate_with_scores(all_scores, all_labels, session_ids=all_session_ids, prices=all_prices)
            result = self.merge_scores(result)
            return result

    def remove_padding_items(self, batch_itemids):
        N = len(batch_itemids)
        res = np.empty(N, dtype=object)
        for i in range(N):
            res[i] = batch_itemids[i][batch_itemids[i] > 0]
        return res

    @torch.no_grad()
    def evaluate_with_full_items(self, data, model, user_history, verbose=0, infer_batch_size=40960):
        ## item_price: [n_items, ], np.array
        ## Attention: currently, only supports predict_layer_type=dot
        model.eval()
        model = self.accelerator.unwrap_model(model)
        iter_data = (
            tqdm(
                enumerate(data),
                total=len(data),
                desc="Evaluate",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process,
            )
            if verbose == 2
            else enumerate(data)
        )

        all_results = []
        item_embeddings = model.forward_all_item_emb(infer_batch_size)
        if self.config["distance_type"] == DistanceType.COSINE.value:
            if not isinstance(item_embeddings, ssp.spmatrix):
                item_embeddings /= np.linalg.norm(item_embeddings) + EPS
        if model.has_item_bias:
            item_bias = model.get_all_item_bias()
            item_bias = item_bias.reshape((1, -1))

        for _, inter_data in iter_data:
            samples = {k: inter_data[v] for k, v in data.dataset.return_key_2_index.items()}
            inputs = {k: v for k, v in samples.items() if k in inspect.signature(model.forward_user_emb).parameters}
            user_embeddings = model.forward_user_emb(**inputs)
            user_embeddings = tensor2array(user_embeddings)
            batch_userids = samples["user_id"]
            batch_itemids = samples["item_id"]
            batch_userids = tensor2array(batch_userids)
            batch_itemids = tensor2array(batch_itemids)
            if data.dataset.config["data_format"] in {
                DataFileFormat.T5.value,
                DataFileFormat.T6.value,
            }:
                batch_itemids = self.remove_padding_items(batch_itemids)

            if isinstance(item_embeddings, ssp.spmatrix):
                # For SAR, the multiply of two sparse matrix is slow.
                # Therefore, a for-loop-based function decorated with numba.jit is used to accelerate.
                batch_scores = model.sparse_matrix_mul(user_embeddings, item_embeddings)
            else:
                if self.config["distance_type"] == DistanceType.DOT.value:
                    batch_scores = user_embeddings @ item_embeddings.T
                elif self.config["distance_type"] == DistanceType.COSINE.value:
                    # batch_scores = sp.distance.cdist(user_embeddings, item_embeddings, 'cosine')
                    user_embeddings /= np.linalg.norm(user_embeddings) + EPS
                    batch_scores = user_embeddings @ item_embeddings.T
                else:
                    raise ValueError("Unsupported distance_type for full item evaluation: {0}".format(self.config["distance_type"]))
            if isinstance(batch_scores, ssp.spmatrix):
                batch_scores = batch_scores.toarray()
            batch_scores = np.array(batch_scores)  # convert some Matrix type to array
            if model.has_item_bias:
                batch_scores += item_bias
            if model.has_user_bias:
                user_bias = model.get_user_bias(samples)
                batch_scores += user_bias.reshape(-1, 1)
            batch_scores = batch_scores / self.config["tau"]

            for idx, userid in enumerate(batch_userids):
                itemid = batch_itemids[idx]
                if data.dataset.config["data_format"] not in {
                    DataFileFormat.T5.value,
                    DataFileFormat.T6.value,
                } and (isinstance(itemid, list) or isinstance(itemid, np.ndarray)):
                    ## only one item is positive
                    itemid = itemid[0]
                target_score = batch_scores[idx, itemid]

                if userid < len(user_history) and user_history[userid] is not None:
                    history = user_history[userid]
                    batch_scores[idx][history] = self.NINF

                if self.__class__.__name__ == "OnePositiveEvaluator":
                    batch_scores[idx][0] = target_score
                    batch_scores[idx][itemid] = self.NINF
                else:
                    batch_scores[idx][0] = self.NINF
                    batch_scores[idx][itemid] = target_score

            # batch_scores = self.accelerator.gather_for_metrics(torch.tensor(batch_scores, device=self.accelerator.device)).cpu().numpy()
            # batch_itemids = self.accelerator.gather_for_metrics(torch.tensor(batch_itemids, device=self.accelerator.device)).cpu().numpy()
            # batch_prices = self.accelerator.gather_for_metrics(torch.tensor(batch_prices, device=self.accelerator.device)).cpu().numpy() if batch_prices is not None else None

            result = self.evaluate_with_scores(batch_scores, pos_itemids=batch_itemids)
            for k, v in result.items():
                result[k] = self.accelerator.gather_for_metrics(torch.tensor(v, device=self.accelerator.device)).cpu().numpy()
            all_results.append(result)

        result = self.merge_scores(all_results)
        return result
