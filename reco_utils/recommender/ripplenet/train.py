# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import tensorflow as tf
import numpy as np
from reco_utils.recommender.ripplenet.model import RippleNet

def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    ripple_set = data_info[5]

    model = RippleNet(args, n_entity, n_relation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # evaluation
            train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)

            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    while start < data.shape[0]:
        auc, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list))
