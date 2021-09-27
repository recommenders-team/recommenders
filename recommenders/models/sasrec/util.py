# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
import copy
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
import time


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    # f = open('data/%s.txt' % fname, 'r')
    f = open(fname, "r")
    for line in f:
        u, i = line.rstrip().split(" ")
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
    f = open("data/%s.txt" % fname, "r")

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

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):

        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])

        # inverse to get descending sort
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, maxlen, num_neg_test):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in tqdm(users, ncols=70, leave=False, unit="b"):
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue

        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(num_neg_test):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        inputs = {}
        inputs["user"] = np.expand_dims(np.array([u]), axis=-1)
        inputs["input_seq"] = np.array([seq])
        inputs["candidate"] = np.array([item_idx])

        # predictions = -model.predict(sess, [u], [seq], item_idx)
        predictions = -1.0 * model.predict(inputs)
        predictions = np.array(predictions)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

    return NDCG / valid_user, HT / valid_user


def create_combined_dataset(u, seq, pos, neg, seq_max_len):
    """
    function to create model inputs from sampled batch data.
    This function is used only during training.
    """
    inputs = {}
    seq = tf.keras.preprocessing.sequence.pad_sequences(
        seq, padding="pre", truncating="pre", maxlen=seq_max_len
    )
    pos = tf.keras.preprocessing.sequence.pad_sequences(
        pos, padding="pre", truncating="pre", maxlen=seq_max_len
    )
    neg = tf.keras.preprocessing.sequence.pad_sequences(
        neg, padding="pre", truncating="pre", maxlen=seq_max_len
    )

    inputs["users"] = np.expand_dims(np.array(u), axis=-1)
    inputs["input_seq"] = seq
    inputs["positive"] = pos
    inputs["negative"] = neg

    target = np.concatenate(
        [
            np.repeat(1, seq.shape[0] * seq.shape[1]),
            np.repeat(0, seq.shape[0] * seq.shape[1]),
        ],
        axis=0,
    )
    target = np.expand_dims(target, axis=-1)
    return inputs, target


def train(model, dataset, sampler, **kwargs):
    """
    High level function for model training
    """
    num_epochs = kwargs.get("num_epochs", 10)
    maxlen = kwargs.get("maxlen", 100)
    num_neg_test = kwargs.get("num_neg_test", 100)
    batch_size = kwargs.get("batch_size", 128)
    lr = kwargs.get("learning_rate", 0.001)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_steps = int(len(user_train) / batch_size)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )

    loss_function = model.loss_function

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    train_step_signature = [
        {
            "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            "input_seq": tf.TensorSpec(shape=(None, maxlen), dtype=tf.int64),
            "positive": tf.TensorSpec(shape=(None, maxlen), dtype=tf.int64),
            "negative": tf.TensorSpec(shape=(None, maxlen), dtype=tf.int64),
        },
        tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        with tf.GradientTape() as tape:
            pos_logits, neg_logits, loss_mask = model(inp, training=True)
            loss = loss_function(pos_logits, neg_logits, loss_mask)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        return loss

    T = 0.0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):

        step_loss = []
        train_loss.reset_states()
        for step in tqdm(
            range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        ):

            u, seq, pos, neg = sampler.next_batch()

            inputs, target = create_combined_dataset(u, seq, pos, neg, maxlen)

            loss = train_step(inputs, target)
            step_loss.append(loss)

        #     print(f"Epoch: {epoch}, Loss: {np.mean(step_loss):.3f}, {train_loss.result():.3f}")

        if epoch % 2 == 0:
            t1 = time.time() - t0
            T += t1
            print("Evaluating...")
            t_test = evaluate(model, dataset, maxlen, num_neg_test)
            t_valid = evaluate_valid(model, dataset, maxlen, num_neg_test)
            print(
                f"\nepoch: {epoch}, time: {T}, valid (NDCG@10: {t_valid[0]}, HR@10: {t_valid[1]})"
            )
            print(
                f"epoch: {epoch}, time: {T},  test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})"
            )
            t0 = time.time()

    t_test = evaluate(model, dataset, maxlen, num_neg_test)
    print(f"\nepoch: {epoch}, test (NDCG@10: {t_test[0]}, HR@10: {t_test[1]})")

    return t_test
