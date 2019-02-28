"""define metrics"""
from collections import defaultdict
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error,mean_absolute_error,r2_score,explained_variance_score
import numpy as np
import utils.util as util


def cal_metric(labels, preds, hparams, topN_flag, flag):
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
        ##added evaluation metric for rating by Ye
        elif metric == 'mae':
            res['mae'] = mean_absolute_error(np.asarray(labels), np.asarray(preds))
        elif metric == 'rsquare':
            res['rsquare'] = r2_score(np.asarray(labels), np.asarray(preds))
        elif metric == 'exp_var':
            res['exp_var'] = explained_variance_score(np.asarray(labels), np.asarray(preds))

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
