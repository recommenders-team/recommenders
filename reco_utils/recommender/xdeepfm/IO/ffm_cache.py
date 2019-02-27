"""define FfmCache class for cache the format dataset"""
from IO.base_cache import BaseCache
import tensorflow as tf
import numpy as np
from collections import defaultdict
import utils.util as util

__all__ = ["FfmCache"]


class FfmCache(BaseCache):
    # field index start by 1, feat index start by 1
    def _load_batch_data_from_file(self, file, hparams):
        batch_size = hparams.batch_size
        labels = []
        features = []
        impression_id = []
        cnt = 0
        with open(file, 'r') as rd:
            while True:
                line = rd.readline().strip(' ')
                if not line:
                    break
                tmp = line.strip().split(util.USER_ID_SPLIT)
                if len(tmp) == 2:
                    impression_id.append(tmp[1].strip())
                line = tmp[0]
                cols = line.strip().split(' ')
                label = float(cols[0].strip())
                if label > 0:
                    label = 1
                else:
                    label = 0
                cur_feature_list = []
                for word in cols[1:]:
                    if not word.strip():
                        continue
                    tokens = word.strip().split(':')
                    cur_feature_list.append( \
                        [int(tokens[0]) - 1, \
                         int(tokens[1]) - 1, \
                         float(tokens[2])])
                features.append(cur_feature_list)
                labels.append(label)
                cnt += 1
                if cnt == batch_size:
                    yield labels, features, impression_id
                    labels = []
                    features = []
                    impression_id = []
                    cnt = 0
        if cnt > 0:
            yield labels, features, impression_id

    def _convert_data(self, labels, features, hparams):
        dim = hparams.FEATURE_COUNT
        FIELD_COUNT = hparams.FIELD_COUNT
        instance_cnt = len(labels)

        fm_feat_indices = []
        fm_feat_values = []
        fm_feat_shape = [instance_cnt, dim]

        dnn_feat_indices = []
        dnn_feat_values = []
        dnn_feat_weights = []
        dnn_feat_shape = [instance_cnt * FIELD_COUNT, -1]

        for i in range(instance_cnt):
            m = len(features[i])
            dnn_feat_dic = {}
            for j in range(m):
                fm_feat_indices.append([i, features[i][j][1]])
                fm_feat_values.append(features[i][j][2])
                if features[i][j][0] not in dnn_feat_dic:
                    dnn_feat_dic[features[i][j][0]] = 0
                else:
                    dnn_feat_dic[features[i][j][0]] += 1
                dnn_feat_indices.append([i * FIELD_COUNT + features[i][j][0], \
                                         dnn_feat_dic[features[i][j][0]]])
                dnn_feat_values.append(features[i][j][1])
                dnn_feat_weights.append(features[i][j][2])
                if dnn_feat_shape[1] < dnn_feat_dic[features[i][j][0]]:
                    dnn_feat_shape[1] = dnn_feat_dic[features[i][j][0]]
        dnn_feat_shape[1] += 1

        sorted_index = sorted(range(len(dnn_feat_indices)),
                              key=lambda k: (dnn_feat_indices[k][0], \
                                             dnn_feat_indices[k][1]))

        res = {}
        res['fm_feat_indices'] = np.asarray(fm_feat_indices, dtype=np.int64)
        res['fm_feat_values'] = np.asarray(fm_feat_values, dtype=np.float32)
        res['fm_feat_shape'] = np.asarray(fm_feat_shape, dtype=np.int64)
        res['labels'] = np.asarray([[label] for label in labels], dtype=np.float32)

        res['dnn_feat_indices'] = np.asarray(dnn_feat_indices, dtype=np.int64)[sorted_index]
        res['dnn_feat_values'] = np.asarray(dnn_feat_values, dtype=np.int64)[sorted_index]
        res['dnn_feat_weights'] = np.asarray(dnn_feat_weights, dtype=np.float32)[sorted_index]
        res['dnn_feat_shape'] = np.asarray(dnn_feat_shape, dtype=np.int64)
        return res

    def write_tfrecord(self, infile, outfile, hparams):
        sample_num = 0
        FEATURE_COUNT = hparams.FEATURE_COUNT
        writer = tf.python_io.TFRecordWriter(outfile)
        feature_cnt = defaultdict(lambda: 0)
        impression_id_list = []
        try:
            for labels, features, impression_id in self._load_batch_data_from_file(infile, hparams):
                impression_id_list.extend(impression_id)
                sample_num += len(labels)
                input_in_sp = self._convert_data(labels, features, hparams)
                fm_feat_indices = input_in_sp['fm_feat_indices']

                for feat in fm_feat_indices:
                    feature_cnt[feat[1]] += 1

                fm_feat_values = input_in_sp['fm_feat_values']
                fm_feat_shape = input_in_sp['fm_feat_shape']
                labels = input_in_sp['labels']
                dnn_feat_indices = input_in_sp['dnn_feat_indices']
                dnn_feat_values = input_in_sp['dnn_feat_values']
                dnn_feat_weights = input_in_sp['dnn_feat_weights']
                dnn_feat_shape = input_in_sp['dnn_feat_shape']

                fm_feat_indices_str = fm_feat_indices.tostring()
                labels_str = labels.tostring()
                dnn_feat_indices_str = dnn_feat_indices.tostring()

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'fm_feat_indices': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[fm_feat_indices_str])),
                            'fm_feat_values': tf.train.Feature(
                                float_list=tf.train.FloatList(value=fm_feat_values)),
                            'fm_feat_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=fm_feat_shape)),
                            'labels': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[labels_str])),
                            'dnn_feat_indices': tf.train.Feature(
                                bytes_list=tf.train.BytesList(value=[dnn_feat_indices_str])),
                            'dnn_feat_values': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=dnn_feat_values)),
                            'dnn_feat_weights': tf.train.Feature(
                                float_list=tf.train.FloatList(value=dnn_feat_weights)),
                            'dnn_feat_shape': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=dnn_feat_shape))
                        }
                    )
                )
                serialized = example.SerializeToString()
                writer.write(serialized)
        except:
            raise ValueError('train data format must be libffm, for example 1 2:1:0.1 2:3:0.2 3:4:0.4')
        writer.close()
        sort_feature_cnt = sorted(feature_cnt.items(), key=lambda x: x[0])
        with open(util.FEAT_COUNT_FILE, 'w') as f:
            for item in sort_feature_cnt:
                f.write(str(item[0]) + ',' + str(item[1]) + '\n')
        return sample_num, impression_id_list
