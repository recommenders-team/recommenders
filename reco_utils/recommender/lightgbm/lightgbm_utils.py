# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os, logging
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
import pandas as pd
import category_encoders as ce
from tqdm import tqdm
import collections
import gc

import tarfile
from reco_utils.dataset.url_utils import maybe_download

logging.basicConfig(level = logging.INFO, format = '%(asctime)s [INFO] %(message)s')

def download_lgb_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    tar_ref = tarfile.open(os.path.join(data_path, remote_resource_name), "r")
    tar_ref.extractall(data_path)
    tar_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))

def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss
    FIXME: refactor this with the reco metrics
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res

def unpackbits(x,num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

class NumEncoder(object):
    def __init__(self, cate_cols, nume_cols, label_col, threshold=10, thresrate=0.99):
        self.label_name = label_col
        self.cate_cols = cate_cols
        self.dtype_dict = {}
        for item in cate_cols:
            self.dtype_dict[item] = 'str'
        for item in nume_cols:
            self.dtype_dict[item] = 'float'
        self.nume_cols = nume_cols
        self.tgt_nume_cols = []
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_cols)
        self.threshold = threshold
        self.thresrate = thresrate
        
        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.Max_len = {}
        self.samples = 0

    def fit_transform(self, inPath):
        logging.info('----------------------------------------------------------------------')
        logging.info('Fitting and Transforming %s .'%inPath)
        logging.info('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        self.samples = df.shape[0]
        logging.info('Filtering and fillna features')
        for item in tqdm(self.cate_cols):
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(value_counts[:int(num*self.thresrate)][value_counts>self.threshold].index)
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')
            del value_counts
            gc.collect()

        for item in tqdm(self.nume_cols):
            df[item] = df[item].fillna(df[item].mean())
            self.save_num_embs[item] = {'sum':df[item].sum(), 'cnt':df[item].shape[0]}
        

        logging.info('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)

        logging.info('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean':[], 'count':[]}
            feat_temp_result = collections.defaultdict(lambda : [0, 0])
            self.save_cate_avgs[item] = collections.defaultdict(lambda : [0, 0])
            for idx in range(self.samples):
                cur_feat = feats[idx]
                if cur_feat in self.save_cate_avgs[item]:
                    feat_encoding['mean'].append(self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1])
                    feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1]/idx)
                else:
                    feat_encoding['mean'].append(0)
                    feat_encoding['count'].append(0)
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item+'_t_mean'] = feat_encoding['mean']
            df[item+'_t_count'] = feat_encoding['count']
            self.tgt_nume_cols.append(item+'_t_mean')
            self.tgt_nume_cols.append(item+'_t_count')
        
        logging.info('Start manual binary encoding')
        rows = None
        for item in tqdm(self.nume_cols+self.tgt_nume_cols):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            Max = df[item].max()
            bit_len = len(bin(Max)) - 2
            samples = self.samples
            self.Max_len[item] = bit_len
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        trn_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        trn_x = np.array(rows)
        return (trn_x, trn_y)

    # for test dataset
    def transform(self, inPath):
        logging.info('----------------------------------------------------------------------')
        logging.info('Transforming %s .'%inPath)
        logging.info('----------------------------------------------------------------------')
        df = pd.read_csv(inPath, dtype=self.dtype_dict)
        samples = df.shape[0]
        logging.info('Filtering and fillna features')
        for item in tqdm(self.cate_cols):
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index)-set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in tqdm(self.nume_cols):
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        logging.info('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)

        logging.info('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in tqdm(self.cate_cols):
            avgs = self.save_cate_avgs[item]
            df[item+'_t_mean'] = df[item].map(lambda x: avgs[x][0]/avgs[x][1] if x in avgs else 0)
            df[item+'_t_count'] = df[item].map(lambda x: avgs[x][1]/self.samples if x in avgs else 0)
        
        logging.info('Start manual binary encoding')
        rows = None
        for item in tqdm(self.nume_cols+self.tgt_nume_cols):
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1,1))
            else:
                rows = np.concatenate([rows,feats.reshape((-1,1))],axis=1)
            del feats
            gc.collect()
        for item in tqdm(self.cate_cols):
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples,-1))
            rows = np.concatenate([rows,res],axis=1)
            del feats
            gc.collect()
        vld_y = np.array(df[self.label_name].values).reshape((-1,1))
        del df
        gc.collect()
        vld_x = np.array(rows)
        return (vld_x, vld_y)
