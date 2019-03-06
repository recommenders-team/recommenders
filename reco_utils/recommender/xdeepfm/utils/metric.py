"""define metrics"""
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
import numpy as np
import utils.util as util


def cal_metric(labels, preds, hparams, flag):
    """Calculate metrics,such as auc, logloss, group auc"""
    res = {}

    def load_impression_id(file_name):
        """load impression id, such as user id, news id"""
        id_list = []
        with open(file_name, 'r') as f_in:
            for line in f_in:
                id_list.append(line.strip())
        return id_list

    for metric in hparams.metrics:
        if metric == 'auc':
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res['auc'] = round(auc, 4)
        elif metric == 'rmse':
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res['rmse'] = np.sqrt(round(rmse, 4))
        elif metric == 'logloss':
            # avoid logloss nan
            preds = [max(min(p, 1. - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res['logloss'] = round(logloss, 4)
        elif metric == 'group_auc':
            if flag == 'train':
                impression_id_list = load_impression_id(util.TRAIN_IMPRESSION_ID)
                if len(impression_id_list) == 0:
                    raise ValueError("train data does not has impressionId," \
                                     "so can not cal the group auc!")
                group_auc = cal_group_auc(labels, preds, impression_id_list)
                res['group_auc'] = group_auc
            elif flag == 'eval':
                impression_id_list = load_impression_id(util.EVAL_IMPRESSION_ID)
                if len(impression_id_list) == 0:
                    raise ValueError("eval data does not has impressionId," \
                                     "so can not cal the group auc!")
                group_auc = cal_group_auc(labels, preds, impression_id_list)
                res['group_auc'] = group_auc
            elif flag == 'test':
                impression_id_list = load_impression_id(util.INFER_IMPRESSION_ID)
                if len(impression_id_list) == 0:
                    raise ValueError("infer data does not has impressionId," \
                                     "so can not cal the group auc!")
                group_auc = cal_group_auc(labels, preds, impression_id_list)
                res['group_auc'] = group_auc
            else:
                raise ValueError("cal metric dataSet should be train, eval , test")

        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def cal_group_auc(labels, preds, impression_id_list):
    """Calculate group auc"""
    if len(impression_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(impression_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = impression_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(impression_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc
    
def recommend_k_items(self, test, top_k=10, sort_top_k=False, **kwargs):
    """Recommend top K items for all users which are in the test set

    Args:
        **kwargs:

    Returns:
        pd.DataFrame: A DataFrame that contains top k recommendation items for each user.
    """

    # pick users from test set and
    test_users = test[self.col_user].unique()
    try:
        test_users_training_ids = np.array(
            [self.user_map_dict[user] for user in test_users]
        )
    except KeyError():
        msg = "SAR cannot score test set users which are not in the training set"
        log.error(msg)
        raise ValueError(msg)

    # shorthand
    scores = self.scores

    # Convert to dense, the following operations are easier.
    log.info("Converting to dense matrix...")
    if isinstance(scores, np.matrixlib.defmatrix.matrix):
        scores_dense = np.array(scores)
    else:
        scores_dense = scores.todense()

    # take the intersection between train test items and items we actually need
    test[self._col_hashed_users] = test[self.col_user].map(self.user_map_dict)

    # Mask out items in the train set.  This only makes sense for some
    # problems (where a user wouldn't interact with an item more than once).
    if self.remove_seen:
        log.info("Removing seen items...")
        scores_dense[self.index[:, 0], self.index[:, 1]] = 0

    # Get top K items and scores.
    log.info("Getting top K...")
    top_items = np.argpartition(scores_dense, -top_k, axis=1)[:, -top_k:]
    top_scores = scores_dense[np.arange(scores_dense.shape[0])[:, None], top_items]

    log.info("Select users from the test set")
    top_items = top_items[test_users_training_ids, :]
    top_scores = top_scores[test_users_training_ids, :]

    log.info("Creating output dataframe...")

    # Convert to np.array (from view) and flatten
    top_items = np.reshape(np.array(top_items), -1)
    top_scores = np.reshape(np.array(top_scores), -1)

    userids = []
    for u in test_users:
        userids.extend([u] * top_k)

    results = pd.DataFrame.from_dict(
        {
            self.col_user: userids,
            self.col_item: top_items,
            self.col_rating: top_scores,
        }
    )

    # remap user and item indices to IDs
    results[self.col_item] = results[self.col_item].map(self.index2item)

    # do final sort
    if sort_top_k:
        results = (
            results.sort_values(
                by=[self.col_user, self.col_rating], ascending=False
            )
            .groupby(self.col_user)
            .apply(lambda x: x)
        )

    # format the dataframe in the end to conform to Suprise return type
    log.info("Formatting output")

    # modify test to make it compatible with

    return (
        results[[self.col_user, self.col_item, self.col_rating]]
        .rename(columns={self.col_rating: PREDICTION_COL})
        .astype(
            {
                self.col_user: _user_item_return_type(),
                self.col_item: _user_item_return_type(),
                PREDICTION_COL: _predict_column_type(),
            }
        )
    )
