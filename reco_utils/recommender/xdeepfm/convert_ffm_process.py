# -*- coding: utf-8 -*-
import sys, csv, math
from collections import defaultdict
import random
# train_file = './toy_data/train_kaggle_ctr_toy.txt'
# test_file = './toy_data/test_kaggle_ctr_toy.txt'

# train_file = './300w_train.txt'
# eval_file = './300w_eval.txt'
# test_file = './300w_test.txt'

def sample(input_path, train_path, eval_path, test_path, ratio=0.8):
    train_out = open(train_path, 'w')
    eval_out = open(eval_path, 'w')
    test_out = open(test_path, 'w')
    for index, line in enumerate(open(input_path, 'r')):
        if index == 0:
            train_out.write(line)
            eval_out.write(line)
            test_out.write(line)	
            continue
        p = random.random()
        if p < ratio:
            train_out.write(line)
        elif p < ratio + 0.1:
            eval_out.write(line)
        else:
            test_out.write(line)
    train_out.close()
    eval_out.close()
    test_out.close()
"""
sample(input_path, train_file, eval_file)

# 统计category feat的词频
feat_cnt = defaultdict(lambda: 0)
"""
def scan(filename, feat_cnt):
    for row in csv.DictReader(open(filename)):
        for key, val in row.items():
            if 'C' in key:
                if val == '':
                    feat_cnt[str(key) + '#' + 'absence'] += 1
                else:
                    feat_cnt[str(key) + '#' + str(val)] += 1
"""
scan(train_file)
scan(eval_file)
scan(test_file)

# 离散特征判断为长尾的阈值
T = 4 
"""
#考虑连续特征离散化和长尾特征之后，统计训练集和测试集中的feat
def get_feat(featSet, filename):
    for row in csv.DictReader(open(filename)):
        for key, val in row.items():
            if 'I' in key and key != "Id":
                if val == '':
                    featSet.add(str(key) + '#' + 'absence')
                else:
                    val = int(val)
                    if val > 2:
                        val = int(math.log(float(val)) ** 2)
                    else:
                        val = 'SP' + str(val)
                    featSet.add(str(key) + '#' + str(val))
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > T:
                    featSet.add(feat)
                else:
                    featSet.add(str(key) + '#' + str(feat_cnt[feat]))
                continue
"""
featSet = set()
get_feat(featSet, train_file)
get_feat(featSet, test_file)
get_feat(featSet, eval_file)


print('train and test data total feat num:', len(featSet))

featIndex = dict()
for index, feat in enumerate(featSet, start=1):
    featIndex[feat] = index
    # print(index, feat)
print('feat dict num:', len(featIndex))

fieldIndex = dict()
fieldList = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', \
             'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', \
             'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

for index, field in enumerate(fieldList, start=1):
    fieldIndex[field] = index
print('field dict num:', len(fieldIndex))
"""

def convert_to_ffm(src_path, dst_path, fieldIndex, featIndex):
    out = open(dst_path, 'w')
    for row in csv.DictReader(open(src_path)):
        feats = []
        feats.append(row['Label'])
        for key, val in row.items():
            if key == 'Label':
                continue
            if 'I' in key and key != "Id":
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    val = int(val)
                    if val > 2:
                        val = int(math.log(float(val)) ** 2)
                    else:
                        val = 'SP' + str(val)
                    feat = str(key) + '#' + str(val)
                feats.append(str(fieldIndex[key]) + ':' + str(featIndex[feat]) + ':1')
                continue
            if 'C' in key:
                if val == '':
                    feat = str(key) + '#' + 'absence'
                else:
                    feat = str(key) + '#' + str(val)
                if feat_cnt[feat] > T:
                    feat = feat
                else:
                    feat = str(key) + '#' + str(feat_cnt[feat])
                feats.append(str(fieldIndex[key]) + ':' + str(featIndex[feat]) + ':1')
                continue
        out.write(' '.join(feats) + '\n')
    out.close()

"""
tr_src = './300w_train.txt'
tr_dst = './300w_train_T4.process.new.ffm'
convert_to_ffm(tr_src, tr_dst, featIndex)

tr_src = './300w_eval.txt'
tr_dst = './300w_eval_T4.process.new.ffm'
convert_to_ffm(tr_src, tr_dst, featIndex)

tr_src = './300w_test.txt'
tr_dst = './300w_test_T4.process.new.ffm'
convert_to_ffm(tr_src, tr_dst, featIndex)
"""
if __name__ == '__main__':
    input_path = "../input.csv"
    train_file = "../train.csv"
    eval_file = "../eval.csv"
    test_file = "../test.csv"
    sample(input_path, train_file, eval_file, test_file)
    # 统计category feat的词频
    feat_cnt = defaultdict(lambda: 0)
    scan(train_file, feat_cnt)
    scan(eval_file, feat_cnt)
    scan(test_file, feat_cnt)
    # 离散特征判断为长尾的阈值
    T = 4
    featSet = set()
    get_feat(featSet, train_file)
    get_feat(featSet, test_file)
    get_feat(featSet, eval_file)

    print('train and test data total feat num:', len(featSet))

    featIndex = dict()
    for index, feat in enumerate(featSet, start=1):
        featIndex[feat] = index
        # print(index, feat)
    print('feat dict num:', len(featIndex))

    fieldIndex = dict()
    fieldList = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1', 'C2', 'C3',
                 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
                 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

    for index, field in enumerate(fieldList, start=1):
        fieldIndex[field] = index
    print('field dict num:', len(fieldIndex))
    
    convert_to_ffm(train_file, "../train.ffm", fieldIndex, featIndex)
    convert_to_ffm(eval_file, "../eval.ffm", fieldIndex, featIndex)
    convert_to_ffm(test_file, "../test.ffm", fieldIndex, featIndex)
