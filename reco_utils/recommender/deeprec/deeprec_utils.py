
"""This script parse and run train function"""

import tensorflow as tf
import six
import os
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score, f1_score
import numpy as np
import yaml


def flat_config(config):
    """flat config to a dict"""
    f_config = {}
    category = ['data', 'model', 'train', 'info']
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config



def check_file_exist(filename):
    if not os.path.isfile(filename):
        raise ValueError("{0} is not exits".format(filename))


def load_yaml_file(filename):
    with open(filename) as f:
        try:
            config = yaml.load(f)
        except:
            raise IOError("load {0} error!".format(filename))
    return config


# train process load yaml
def load_yaml(yaml_name):
    check_file_exist(yaml_name)
    config = load_yaml_file(yaml_name)
    return config



def create_hparams(FLAGS):
    return tf.contrib.training.HParams(
        # data
        kg_file=FLAGS['kg_file'] if 'kg_file' in FLAGS else None,
        user_clicks=FLAGS['user_clicks'] if 'user_clicks' in FLAGS else None,
        FEATURE_COUNT=FLAGS['FEATURE_COUNT'] if 'FEATURE_COUNT' in FLAGS else None,
        FIELD_COUNT=FLAGS['FIELD_COUNT'] if 'FIELD_COUNT' in FLAGS else None,
        data_format=FLAGS['data_format'] if 'data_format' in FLAGS else None,
        PAIR_NUM=FLAGS['PAIR_NUM'] if 'PAIR_NUM' in FLAGS else None,
        DNN_FIELD_NUM=FLAGS['DNN_FIELD_NUM'] if 'DNN_FIELD_NUM' in FLAGS else None,
        n_user=FLAGS['n_user'] if 'n_user' in FLAGS else None,
        n_item=FLAGS['n_item'] if 'n_item' in FLAGS else None,
        n_user_attr=FLAGS['n_user_attr'] if 'n_user_attr' in FLAGS else None,
        n_item_attr=FLAGS['n_item_attr'] if 'n_item_attr' in FLAGS else None,
        iterator_type=FLAGS['iterator_type'] if 'iterator_type' in FLAGS else None,
        SUMMARIES_DIR=FLAGS['SUMMARIES_DIR'] if 'SUMMARIES_DIR' in FLAGS else None,
        MODEL_DIR=FLAGS['MODEL_DIR'] if 'MODEL_DIR' in FLAGS else None,

        ### ripple
        n_entity = FLAGS['n_entity'] if 'n_entity' in FLAGS else None,
        n_memory = FLAGS['n_memory'] if 'n_memory' in FLAGS else None,
        n_relation = FLAGS['n_relation'] if 'n_relation' in FLAGS else None,
        n_users = FLAGS['n_users'] if 'n_users' in FLAGS else None,
        n_items = FLAGS['n_items'] if 'n_items' in FLAGS else None,
        entity_limit = FLAGS['entity_limit'] if 'entity_limit' in FLAGS else None,
        user_click_limit = FLAGS['user_click_limit'] if 'user_click_limit' in FLAGS else None,
        # dkn
        doc_size=FLAGS['doc_size'] if 'doc_size' in FLAGS else None,
        word_size=FLAGS['word_size'] if 'word_size' in FLAGS else None,
        entity_size=FLAGS['entity_size'] if 'entity_size' in FLAGS else None,
        entity_dim=FLAGS['entity_dim'] if 'entity_dim' in FLAGS else None,
        entity_embedding_method=FLAGS['entity_embedding_method']
        if 'entity_embedding_method' in FLAGS else None,
        transform=FLAGS['transform'] if 'transform' in FLAGS else None,
        train_ratio=FLAGS['train_ratio'] if 'train_ratio' in FLAGS else None,

        # model
        dim=FLAGS['dim'] if 'dim' in FLAGS else None,
        layer_sizes=FLAGS['layer_sizes'] if 'layer_sizes' in FLAGS else None,
        cross_layer_sizes=FLAGS['cross_layer_sizes'] if 'cross_layer_sizes' in FLAGS else None,
        cross_layers=FLAGS['cross_layers'] if 'cross_layers' in FLAGS else None,
        activation=FLAGS['activation'] if 'activation' in FLAGS else None,
        cross_activation=FLAGS['cross_activation'] if 'cross_activation' in FLAGS else "identity",
        user_dropout=FLAGS['user_dropout'] if 'user_dropout' in FLAGS else False,
        dropout=FLAGS['dropout'] if 'dropout' in FLAGS else [0.0],
        attention_layer_sizes=FLAGS['attention_layer_sizes'] if 'attention_layer_sizes' in FLAGS else None,
        attention_activation=FLAGS['attention_activation'] if 'attention_activation' in FLAGS else None,
        attention_dropout=FLAGS['attention_dropout'] \
            if 'attention_dropout' in FLAGS else 0.0,
        model_type=FLAGS['model_type'] if 'model_type' in FLAGS else None,
        method=FLAGS['method'] if 'method' in FLAGS else None,
        load_saved_model=FLAGS['load_saved_model'] if 'load_saved_model' in FLAGS else False,
        load_model_name= FLAGS['load_model_name'] if 'load_model_name' in FLAGS else None,
        filter_sizes=FLAGS['filter_sizes'] if 'filter_sizes' in FLAGS else None,
        num_filters=FLAGS['num_filters'] if 'num_filters' in FLAGS else None,
        mu=FLAGS['mu'] if 'mu' in FLAGS else None,
        fast_CIN_d=FLAGS['fast_CIN_d'] if 'fast_CIN_d' in FLAGS else 0,
        user_Linear_part=FLAGS['user_Linear_part'] if 'user_Linear_part' in FLAGS else False,
        use_FM_part=FLAGS['use_FM_part'] if 'use_FM_part' in FLAGS else False,
        use_CIN_part=FLAGS['use_CIN_part'] if 'use_CIN_part' in FLAGS else False,
        use_DNN_part=FLAGS['use_DNN_part'] if 'use_DNN_part' in FLAGS else False,

        ###ripple
        is_use_relation = FLAGS["is_use_relation"] if "is_use_relation" in FLAGS else False,
        n_entity_emb=FLAGS["n_entity_emb"] if "n_entity_emb" in FLAGS else None,
        n_relation_emb=FLAGS["n_relation_emb"] if "n_relation_emb" in FLAGS else None,
        n_map_emb=FLAGS["n_map_emb"] if "n_map_emb" in FLAGS else None,
        n_hops=FLAGS["n_hops"] if "n_hops" in FLAGS else None,
        item_update_mode=FLAGS["update_item_embedding"] if "update_item_embedding" in FLAGS else None,
        predict_mode=FLAGS["predict_mode"] if "predict_mode" in FLAGS else None,
        n_DCN_layer=FLAGS["n_DCN_layer"] if "n_DCN_layer" in FLAGS else None,
        is_map_feature=FLAGS["is_map_feature"] if "is_map_feature" in FLAGS else False,
        kg_ratio=FLAGS["kg_ratio"] if "kg_ratio" in FLAGS else 1.0,
        output_using_all_hops =FLAGS["output_using_all_hops"] if "output_using_all_hops" in FLAGS else False,
        enable_BN=FLAGS['enable_BN'] if 'enable_BN' in FLAGS else False,

        # train
        init_method=FLAGS['init_method'] if 'init_method' in FLAGS else 'tnormal',
        init_value=FLAGS['init_value'] if 'init_value' in FLAGS else 0.01,
        embed_l2=FLAGS['embed_l2'] if 'embed_l2' in FLAGS else 0.0000,
        embed_l1=FLAGS['embed_l1'] if 'embed_l1' in FLAGS else 0.0000,
        layer_l2=FLAGS['layer_l2'] if 'layer_l2' in FLAGS else 0.0000,
        layer_l1=FLAGS['layer_l1'] if 'layer_l1' in FLAGS else 0.0000,
        cross_l2=FLAGS['cross_l2'] if 'cross_l2' in FLAGS else 0.0000,
        cross_l1=FLAGS['cross_l1'] if 'cross_l1' in FLAGS else 0.0000,
        reg_kg=FLAGS["reg_kg"] if "reg_kg" in FLAGS else 0.0000,
        learning_rate=FLAGS['learning_rate'] if 'learning_rate' in FLAGS else 0.001,
        lr_rs=FLAGS["lr_rs"] if 'lr_rs' in FLAGS else 1,
        lr_kg=FLAGS["lr_kg"] if 'lr_kg' in FLAGS else 0.5,
        kg_training_interval=FLAGS["kg_training_interval"] if 'kg_training_interval' in FLAGS else 5,
        max_grad_norm = FLAGS['max_grad_norm'] if 'max_grad_norm' in FLAGS else 2,
        is_clip_norm = FLAGS['is_clip_norm'] if 'is_clip_norm' in FLAGS else 0,
        dtype = FLAGS['dtype'] if 'dtype' in FLAGS else 32,
        loss=FLAGS['loss'] if 'loss' in FLAGS else None,
        optimizer=FLAGS['optimizer'] if 'optimizer' in FLAGS else 'adam',
        epochs=FLAGS['epochs'] if 'epochs' in FLAGS else 10,
        batch_size=FLAGS['batch_size'] if 'batch_size' in FLAGS else 1,

        # show info
        show_step=FLAGS['show_step'] if 'show_step' in FLAGS else 1,
        save_model=FLAGS['save_model'] if 'save_model' in FLAGS else True,
        save_epoch=FLAGS['save_epoch'] if 'save_epoch' in FLAGS else 5,
        metrics=FLAGS['metrics'] if 'metrics' in FLAGS else None,
        write_tfevents=FLAGS['write_tfevents'] if 'write_tfevents' in FLAGS else False
    )


def prepare_hparams(yaml_file=None, **kwargs):
    if yaml_file:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    if kwargs:
        for name, value in six.iteritems(kwargs):
            config[name] = value

    hparams = create_hparams(config)
    return hparams


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss"""
    res = {}
    for metric in metrics:
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
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res