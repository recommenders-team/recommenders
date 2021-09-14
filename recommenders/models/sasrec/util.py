import sys
import copy
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    # f = open('data/%s.txt' % fname, 'r')
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


def data_partition_with_time(fname, sep=" "):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    Items = set()
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    
    for line in f:
        u, i, t = line.rstrip().split(sep)
        User[u].append((i, t))
        Items.add(i)

    for user in User.keys():
        # sort by time
        items = sorted(User[user], key=lambda x: x[1])
        # keep only the items
        items = [x[0] for x in items]
        User[user] = items
        nfeedback = len(User[user])
        if nfeedback == 1:
            del User[user]
            continue
        elif nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    usernum = len(User)
    itemnum = len(Items)
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, maxlen, num_neg_test):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in tqdm(users, ncols=70, leave=False, unit='b'):

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        # print(train[u])
        # print(valid[u])
        # print(test[u])

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs['user'] = np.expand_dims(np.array([u]), axis=-1)
        inputs['input_seq'] = np.array([seq])
        inputs['candidate'] = np.array([item_idx])

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, maxlen, num_neg_test):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit='b'):
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs['user'] = np.expand_dims(np.array([u]), axis=-1)
        inputs['input_seq'] = np.array([seq])
        inputs['candidate'] = np.array([item_idx])

        # predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        # if valid_user % 100 == 0:
        #     print('.', end="")
        #     sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
