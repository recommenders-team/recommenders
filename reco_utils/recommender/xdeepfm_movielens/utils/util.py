"""define util function and  global variable"""
import tensorflow as tf
import os, sys
import time, yaml
from packaging import version

RES_DIR = './res/'
CACHE_DIR = './cache/'
MODEL_DIR = './checkpoint/'
CONFIG_DIR = './config/'
TRAIN_YAML = 'network_xdeepFM.yaml'
TRAIN_NUM = './cache/train_num.csv'
EVAL_NUM = './cache/eval_num.csv'
TEST_NUM = './cache/test_num.csv'
INFER_NUM = './cache/infer_num.csv'
LOG_DIR = './logs/'
FEAT_COUNT_FILE = './cache/feat_cnt.csv'
TRAIN_IMPRESSION_ID = './cache/train_impressionId.csv'
EVAL_IMPRESSION_ID = './cache/eval_impressionId.csv'
TEST_IMPRESSION_ID = './cache/test_impressionId.csv'
INFER_IMPRESSION_ID = './cache/infer_impressionId.csv'
SUMMARIES_DIR = './logs/'
# define din format feature
DIN_FORMAT_SPLIT = '#'
# split feature and userid
USER_ID_SPLIT = '%'


def check_and_mkdir():
    def make_dir(DIR):
        if not os.path.exists(DIR):
            os.mkdir(DIR)

    make_dir(RES_DIR)
    make_dir(CACHE_DIR)
    make_dir(MODEL_DIR)
    make_dir(CONFIG_DIR)
    make_dir(LOG_DIR)


def check_tensorflow_version():
    if version.parse(tf.__version__) < version.parse("1.2.0"):
        raise EnvironmentError("Tensorflow version must >= 1.2.0,but version is {0}". \
                               format(tf.__version__))


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


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


def convert_cached_name(file_name, batch_size):
    prefix = CACHE_DIR + 'batch_size_' + str(batch_size) + '_'
    prefix += (file_name.strip().split('/'))[-1]
    train_cache_name = prefix.replace(".txt", ".tfrecord"). \
        replace(".csv", ".tfrecord"). \
        replace(".libsvm", ".tfrecord")
    return train_cache_name


def convert_res_name(file_name):
    prefix = RES_DIR
    inferfile = file_name.split('/')[-1]
    res_name = prefix + inferfile.replace("tfrecord", "res.csv"). \
        replace(".csv", ".tfrecord"). \
        replace(".libsvm", ".tfrecord")
    return res_name
