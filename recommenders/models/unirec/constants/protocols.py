# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.

from enum import Enum


class EvaluationProtocal(Enum):
    OneVSAll = "one_vs_all"
    OneVSK = "one_vs_k"
    LabelAware = "label_aware"
    SessionAware = "session_aware"


class DataFileFormat(Enum):
    # user_id, item_id.  This is the interaction format.
    T1 = "user-item"

    # user_id, item_id, label
    T2 = "user-item-label"

    # user_id, item_id, label, session_id
    T2_1 = "user-item-label-session"

    # user_id, item_id, rating. Sometimes the interaction has some levels, such as rating or frequency.
    T3 = "user-item-rating"

    # user_id, item_id_list, label_list. Group each user's candidate items into one group.
    # This is a more compact format than T2. Positive and negative items are distinguished by labels.
    T4 = "user-item_group-label_group"

    # user_id, item_sequence.  Group user's interaction history into a sequence.
    # This is a compact format for transaction history.
    T5 = "user-item_seq"

    # Each line contains a user_id and his item sequence.
    # Different from T5 which is a two-column format, T5_1 is a one-column format,
    # both user_id and item_seq are sepatated by space.
    # This data format is usually used by 8:1:1 split CF datasets like gowalla and yelp2018
    T5_1 = "user_item_seq"

    # user_id, item_sequence, time_sequence
    T6 = "user-item_seq-time_seq"

    # label, index_list, value_list
    # This data format is only used when data is in libFM data format
    # libFM data example: 1 0:1 3000:1 3200:0.3 5000:0.5
    # T7 data example: 1 [0,1,3200,5000] [1,1,0.3,0.5]
    T7 = "label-index_group-value_group"


class ColNames(Enum):
    USERID = "user_id"
    ITEMID = "item_id"
    ITEMID_GROUP = "item_id_list"
    LABEL = "label"
    LABEL_GROUP = "label_list"
    USER_HISTORY = "item_seq"
    TIME_HISTORY = "time_seq"
    SESSION = "session_id"
    USER_AND_HISTORY = "user_id_item_seq"
    INDEX_GROUP = "index_list"
    VALUE_GROUP = "value_list"


class DatasetType(Enum):
    BaseDataset = "BaseDataset"
    SeqRecDataset = "SeqRecDataset"
    AERecDataset = "AERecDataset"
    RankDataset = "RankDataset"


class HistoryMaskMode(Enum):
    Unorder = "unorder"
    Autoregressive = "autoregressive"


class TaskType(Enum):
    TRAIN = "train"  ## normal training model
    TEST = "test"  ## only do evaluation. Should load a pretrained model.
    INFER = "infer"  ## only do score inference. Should load a pretrained model.


class EdgeNormType(Enum):
    NONE = "none"
    SQRT_DEGREE = "sqrt_degree"
