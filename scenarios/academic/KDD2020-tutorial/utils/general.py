import os
import numpy as np

def add2dict(value, d):
    if value not in d:
        d[value] = len(d)
    return d[value]

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list2string(li, sep=','):
    return sep.join([str(a) for a in li])


def dump_dict_as_txt(d, filename):
    with open(filename, 'w', encoding='utf-8') as wt:
        wt.write('{0}\n'.format(len(d)))
        for k, v in d.items():
            wt.write('{0}\t{1}\n'.format(k, v))



def reparameter_sampling(sample_size, probabilities):
    r'''
    Faster sampling algorithm, gumbel softmax trick.
    :param sample_size:
    :param probabilities:
    :return:  index of sampled items (unbiased)
    '''
    random_values = np.random.uniform(size=probabilities.shape)
    random_values = np.log(-np.log(random_values))  #gives gumbel random variable
    shifted_probabilities = random_values - np.log(probabilities)
    return np.argpartition(shifted_probabilities, sample_size)[:sample_size]

def reverse_dict_list(id2list):
    res = {}
    for key, value in id2list.items():
        for id in value:
            if id not in res:
                res[id] = []
            res[id].append(key)
    return res