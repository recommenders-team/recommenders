# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import argparse
import numpy as np
import pandas as pd

def read_item_index_to_entity_id_file(item_to_entity):
    # file = '../data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    item_index_old2new = dict()
    entity_id2index = dict()
    i = 0
    for index, row in item_to_entity.iterrows():
        item_index = str(row[0])
        satori_id = str(row[1])
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return item_index_old2new, entity_id2index


def convert_rating(ratings, item_index_old2new, threshold):

    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for index, row in ratings.iterrows():
        item_index_old = str(int(row[1]))
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(row[0])

        rating = float(row[2])
        if rating >= threshold:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = []
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.append({"user_index": user_index,
            "item": item,
            "rating": 1})
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.append({"user_index": user_index,
                "item": item,
                "rating": 0})
    ratings_final = pd.DataFrame(writer)
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))
    return ratings_final


def convert_kg(kg, entity_id2index):
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    relation_id2index = dict()

    writer = []

    for index, row in kg.iterrows():
        head_old = str(int(row[0]))
        relation_old = row[1]
        tail_old = str(int(row[2]))

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.append({"head": head,
                        "relation": relation,
                        "tail": tail})

    kg_final = pd.DataFrame(writer)
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)
    return kg_final
