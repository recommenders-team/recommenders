# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import collections
import os
import numpy as np


def load_data(ratings_final, kg_final, n_hop, n_memory):
    train_data, eval_data, test_data, user_history_dict = dataset_split(ratings_final)
    n_entity, n_relation, kg = load_kg(kg_final)
    ripple_set = get_ripple_set(kg, user_history_dict, n_hop, n_memory)
    return train_data, eval_data, test_data, n_entity, n_relation, ripple_set

def dataset_split(rating_np):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    # print(len(train_indices), len(eval_indices), len(test_indices))

    # traverse training data, only keeping the users with positive ratings
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np.iloc[i][0]
        item = rating_np.iloc[i][1]
        rating = rating_np.iloc[i][2]
        if rating == 1:
            if user not in user_history_dict:
                user_history_dict[user] = []
            user_history_dict[user].append(item)

    train_indices = [i for i in train_indices if rating_np.iloc[i][0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np.iloc[i][0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np.iloc[i][0] in user_history_dict]
    # print(len(train_indices), len(eval_indices), len(test_indices))

    train_data = rating_np.iloc[train_indices]
    eval_data = rating_np.iloc[eval_indices]
    test_data = rating_np.iloc[test_indices]

    return train_data, eval_data, test_data, user_history_dict


def load_kg(kg_final):
    print('reading KG file ...')

    n_entity = len(set(kg_final.iloc[:, 0]) | set(kg_final.iloc[:, 2]))
    n_relation = len(set(kg_final.iloc[:, 1]))

    kg = construct_kg(kg_final)

    return n_entity, n_relation, kg


def construct_kg(kg_np):
    print('constructing knowledge graph ...')
    kg = collections.defaultdict(list)
    for index, row in kg_np.iterrows():
        kg[row["head"]].append((row["tail"], row["relation"]))
    return kg


def get_ripple_set(kg, user_history_dict, n_hop, n_memory):
    print('constructing ripple set ...')

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < n_memory
                indices = np.random.choice(len(memories_h), size=n_memory, replace=replace)
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
