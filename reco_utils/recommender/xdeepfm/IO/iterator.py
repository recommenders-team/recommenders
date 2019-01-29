"""define iterator"""
import collections
import tensorflow as tf
import abc

BUFFER_SIZE = 256
__all__ = ["BaseIterator", "FfmIterator", "DinIterator", "CCCFNetIterator"]


class BaseIterator(object):
    @abc.abstractmethod
    def get_iterator(self, src_dataset):
        """Subclass must implement this."""
        pass

    @abc.abstractmethod
    def parser(self, record):
        pass


class FfmIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _fm_feat_indices, _fm_feat_values, \
        _fm_feat_shape, _labels, _dnn_feat_indices, \
        _dnn_feat_values, _dnn_feat_weights, _dnn_feat_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_values = _fm_feat_values
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weights = _dnn_feat_weights
        self.dnn_feat_shape = _dnn_feat_shape

    def parser(self, record):
        keys_to_features = {
            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_values': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),
            'labels': tf.FixedLenFeature([], tf.string),
            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weights': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], tf.int64), [-1, 2])
        fm_feat_values = tf.sparse_tensor_to_dense(parsed['fm_feat_values'])
        fm_feat_shape = parsed['fm_feat_shape']
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weights = tf.sparse_tensor_to_dense(parsed['dnn_feat_weights'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return fm_feat_indices, fm_feat_values, \
               fm_feat_shape, labels, dnn_feat_indices, \
               dnn_feat_values, dnn_feat_weights, dnn_feat_shape


class DinIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        output = iterator.get_next()
        (_attention_news_indices, _attention_news_values, _attention_news_shape, \
         _attention_user_indices, _attention_user_values, _attention_user_weights, \
         _attention_user_shape, _fm_feat_indices, _fm_feat_val, \
         _fm_feat_shape, _labels, _dnn_feat_indices, _dnn_feat_values, \
         _dnn_feat_weight, _dnn_feat_shape) = output
        self.initializer = iterator.initializer
        self.attention_news_indices = _attention_news_indices
        self.attention_news_values = _attention_news_values
        self.attention_news_shape = _attention_news_shape
        self.attention_user_indices = _attention_user_indices
        self.attention_user_values = _attention_user_values
        self.attention_user_weights = _attention_user_weights
        self.attention_user_shape = _attention_user_shape
        self.fm_feat_indices = _fm_feat_indices
        self.fm_feat_val = _fm_feat_val
        self.fm_feat_shape = _fm_feat_shape
        self.labels = _labels
        self.dnn_feat_indices = _dnn_feat_indices
        self.dnn_feat_values = _dnn_feat_values
        self.dnn_feat_weight = _dnn_feat_weight
        self.dnn_feat_shape = _dnn_feat_shape

    def parser(self, record):
        keys_to_features = {
            'attention_news_indices': tf.FixedLenFeature([], tf.string),
            'attention_news_values': tf.VarLenFeature(tf.float32),
            'attention_news_shape': tf.FixedLenFeature([2], tf.int64),

            'attention_user_indices': tf.FixedLenFeature([], tf.string),
            'attention_user_values': tf.VarLenFeature(tf.int64),
            'attention_user_weights': tf.VarLenFeature(tf.float32),
            'attention_user_shape': tf.FixedLenFeature([2], tf.int64),

            'fm_feat_indices': tf.FixedLenFeature([], tf.string),
            'fm_feat_val': tf.VarLenFeature(tf.float32),
            'fm_feat_shape': tf.FixedLenFeature([2], tf.int64),

            'labels': tf.FixedLenFeature([], tf.string),

            'dnn_feat_indices': tf.FixedLenFeature([], tf.string),
            'dnn_feat_values': tf.VarLenFeature(tf.int64),
            'dnn_feat_weight': tf.VarLenFeature(tf.float32),
            'dnn_feat_shape': tf.FixedLenFeature([2], tf.int64),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        attention_news_indices = tf.reshape(tf.decode_raw(parsed['attention_news_indices'], \
                                                          tf.int64), [-1, 2])
        attention_news_values = tf.sparse_tensor_to_dense(parsed['attention_news_values'])
        attention_news_shape = parsed['attention_news_shape']

        attention_user_indices = tf.reshape(tf.decode_raw(parsed['attention_user_indices'], \
                                                          tf.int64), [-1, 2])
        attention_user_values = tf.sparse_tensor_to_dense(parsed['attention_user_values'])
        attention_user_weights = tf.sparse_tensor_to_dense(parsed['attention_user_weights'])
        attention_user_shape = parsed['attention_user_shape']

        fm_feat_indices = tf.reshape(tf.decode_raw(parsed['fm_feat_indices'], \
                                                   tf.int64), [-1, 2])
        fm_feat_val = tf.sparse_tensor_to_dense(parsed['fm_feat_val'])
        fm_feat_shape = parsed['fm_feat_shape']

        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])

        dnn_feat_indices = tf.reshape(tf.decode_raw(parsed['dnn_feat_indices'], \
                                                    tf.int64), [-1, 2])
        dnn_feat_values = tf.sparse_tensor_to_dense(parsed['dnn_feat_values'])
        dnn_feat_weight = tf.sparse_tensor_to_dense(parsed['dnn_feat_weight'])
        dnn_feat_shape = parsed['dnn_feat_shape']
        return (attention_news_indices, attention_news_values, attention_news_shape, \
                attention_user_indices, attention_user_values, attention_user_weights, \
                attention_user_shape, fm_feat_indices, fm_feat_val, \
                fm_feat_shape, labels, dnn_feat_indices, dnn_feat_values, \
                dnn_feat_weight, dnn_feat_shape)


class CCCFNetIterator(BaseIterator):
    def __init__(self, src_dataset):
        self.get_iterator(src_dataset)

    def get_iterator(self, src_dataset):
        src_dataset = src_dataset.map(self.parser)
        # src_dataset = src_dataset.shuffle(buffer_size=BUFFER_SIZE)
        iterator = src_dataset.make_initializable_iterator()
        _labels, _userIds, _itemIds, \
        _user_profiles_indices, _user_profiles_values, _user_profiles_weights, _user_profiles_shape, \
        _item_profiles_indices, _item_profiles_values, _item_profiles_weights, _item_profiles_shape = iterator.get_next()
        self.initializer = iterator.initializer
        self.labels = _labels
        self.userIds = _userIds
        self.itemIds = _itemIds
        self.user_profiles_indices = _user_profiles_indices
        self.user_profiles_values = _user_profiles_values
        self.user_profiles_weights = _user_profiles_weights
        self.user_profiles_shape = _user_profiles_shape
        self.item_profiles_indices = _item_profiles_indices
        self.item_profiles_values = _item_profiles_values
        self.item_profiles_weights = _item_profiles_weights
        self.item_profiles_shape = _item_profiles_shape

    def parser(self, record):
        keys_to_features = {
            'labels': tf.FixedLenFeature([], tf.string),
            'userIds': tf.VarLenFeature(tf.int64),
            'itemIds': tf.VarLenFeature(tf.int64),
            'user_profiles_indices': tf.FixedLenFeature([], tf.string),
            'user_profiles_values': tf.VarLenFeature(tf.int64),
            'user_profiles_weights': tf.VarLenFeature(tf.float32),
            'user_profiles_shape': tf.FixedLenFeature([2], tf.int64),
            'item_profiles_indices': tf.FixedLenFeature([], tf.string),
            'item_profiles_values': tf.VarLenFeature(tf.int64),
            'item_profiles_weights': tf.VarLenFeature(tf.float32),
            'item_profiles_shape': tf.FixedLenFeature([2], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        labels = tf.reshape(tf.decode_raw(parsed['labels'], tf.float32), [-1, 1])
        userIds = tf.sparse_tensor_to_dense(parsed['userIds'])
        itemIds = tf.sparse_tensor_to_dense(parsed['itemIds'])

        user_profiles_indices = tf.reshape(tf.decode_raw(parsed['user_profiles_indices'], tf.int64), [-1, 2])
        user_profiles_values = tf.sparse_tensor_to_dense(parsed['user_profiles_values'])
        user_profiles_weights = tf.sparse_tensor_to_dense(parsed['user_profiles_weights'])
        user_profiles_shape = parsed['user_profiles_shape']

        item_profiles_indices = tf.reshape(tf.decode_raw(parsed['item_profiles_indices'], tf.int64), [-1, 2])
        item_profiles_values = tf.sparse_tensor_to_dense(parsed['item_profiles_values'])
        item_profiles_weights = tf.sparse_tensor_to_dense(parsed['item_profiles_weights'])
        item_profiles_shape = parsed['item_profiles_shape']

        return labels, userIds, itemIds, \
               user_profiles_indices, user_profiles_values, user_profiles_weights, user_profiles_shape, \
               item_profiles_indices, item_profiles_values, item_profiles_weights, item_profiles_shape
