r'''
This script contains functions that appear in the tutorial jupyter notebooks (step_1 to step_5).
You can find their usage in the corresponding cells from notebooks.
'''
import codecs
import pickle
import time
import os
from datetime import datetime  
import random
import numpy as np
import math
from multiprocessing import Process

from utils.general import *
from utils.data_helper import *


def gen_paper_content(InFile_PaperTitleAbs_bySentence, OutFileName, word2idx, entity2idx, field=["Title"], doc_len=10):
    if len(word2idx) == 0:
        word2idx['NULL'] = 0
    if len(entity2idx) == 0:
        entity2idx['NULL'] = 0

    paper2content = {}
    print('loading file {0}...'.format(os.path.basename(InFile_PaperTitleAbs_bySentence)))
    with codecs.open(InFile_PaperTitleAbs_bySentence, 'r', 'utf-8') as rd:
        _cnt = 0
        _t0 = time.time()
        while True:
            line = rd.readline()
            if not line:
                break
            _cnt += 1
            if _cnt % 10000 == 0:
                print('\rloading line: {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')
            words = line.strip('\r\n').split('\t')
            paperid, category, position, sentence, fieldOfStudy = words[0], words[1], int(words[2]), words[3], words[4]
            if category not in field:
                continue
            if paperid not in paper2content:
                paper2content[paperid] = []
            if category == "Abstract":
                position += 1000

            words, entities = convert2id(sentence, fieldOfStudy, word2idx, entity2idx)
            paper2content[paperid].append(
                (position, list2string(words, ','), list2string(entities, ','))
            )
    print(' ')

    print('parsing into feature file  ...' )
    with open(OutFileName, 'w') as wt:
        _cnt = 0
        _t0 = time.time()
        for paperid, info in paper2content.items():
            _cnt += 1
            if _cnt % 10000 == 0:
                print('\rparsed paper count: {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')

            words = []
            entities = []
            info.sort(key=lambda x:x[0])
            for clip in info:
                words.extend(clip[1].split(','))
                entities.extend(clip[2].split(','))
            if len(words) > doc_len:
                words = words[0:doc_len]
                entities = entities[0:doc_len]
            elif len(words) < doc_len:
                for _ in range(doc_len - len(words)):
                    words.append('0')
                    entities.append('0')
            wt.write('{0} {1} {2}\n'.format(paperid, ','.join(words), ','.join(entities)))
    print()
    return word2idx, entity2idx


def parse_entities(fieldOfStudy, entity2idx, cnt):
    res = [0] * cnt
    if fieldOfStudy:
        clips = fieldOfStudy.split(',')
        for clip in clips:
            tokens = clip.strip().split(':')
            field_id = tokens[0]
            field_idx = add2dict(field_id, entity2idx)
            start, end = int(tokens[1]), int(tokens[2])
            for i in range(start, end+1):
                res[i] = field_idx
    return res



def convert2id(sentence, fieldOfStudy, word2idx, entity2idx):
    words = sentence.split(' ')
    word_idx = [add2dict(word, word2idx) for word in words]
    entity_idx = parse_entities(fieldOfStudy, entity2idx, len(word_idx))
    return word_idx, entity_idx




def gen_knowledge_relations(InFile_RelatedFieldOfStudy, OutFile_dirname, entity2idx, relation2idx):
    print('processing file {0}...'.format(os.path.basename(InFile_RelatedFieldOfStudy)), end=' ')
    OutFile_relation_triples = os.path.join(OutFile_dirname, 'train2id.txt')
    lines = []
    with open(InFile_RelatedFieldOfStudy, 'r', encoding='utf-8', newline='\r\n') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip('\r\n').split('\t')
            field_idx01 = add2dict(words[0], entity2idx)
            field_idx02 = add2dict(words[2], entity2idx)
            relation_name = '{0}_TO_{1}'.format(words[1], words[3])
            relation_idx = add2dict(relation_name, relation2idx)
            lines.append('{0} {1} {2}\n'.format(field_idx01, field_idx02, relation_idx))
    print('done.')
    with open(OutFile_relation_triples, 'w', encoding='utf-8', newline='\r\n') as wt:
        wt.write('{0}\n'.format(len(lines)))
        for line in lines:
            wt.write(line)
    dump_dict_as_txt(entity2idx, os.path.join(OutFile_dirname, 'entity2id.txt'))
    dump_dict_as_txt(relation2idx, os.path.join(OutFile_dirname, 'relation2id.txt'))


def gen_indexed_sentence_collection(InFile_PaperTitleAbs_bySentence, OutFileName, word2idx):
    print('loading file {0}...'.format(os.path.basename(InFile_PaperTitleAbs_bySentence)))
    with open(InFile_PaperTitleAbs_bySentence, 'r', encoding='utf-8', newline='\r\n') as rd, open(OutFileName, 'w', encoding='utf-8', newline='\r\n') as wt:
        _cnt = 0
        _t0 = time.time()
        while True:
            line = rd.readline()
            if not line:
                break
            _cnt += 1
            if _cnt % 10000 == 0:
                print('\rloading line: {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')
            words = line.strip('\r\n').split('\t')
            paperid, category, position, sentence, fieldOfStudy = words[0], words[1], int(words[2]), words[3], words[4]

            if not sentence:
                continue
            tokens = sentence.split(' ')
            word_idx = [add2dict(token, word2idx) for token in tokens]
            wt.write(list2string(word_idx, ' ') + '\n')

def gen_sentence_collection(InFile_PaperTitleAbs_bySentence, OutFileName, word2idx):
    print('loading file {0}...'.format(os.path.basename(InFile_PaperTitleAbs_bySentence)))
    with open(InFile_PaperTitleAbs_bySentence, 'r', encoding='utf-8', newline='\r\n') as rd, open(OutFileName, 'w', encoding='utf-8', newline='\r\n') as wt:
        _cnt = 0
        _t0 = time.time()
        while True:
            line = rd.readline()
            if not line:
                break
            _cnt += 1
            if _cnt % 10000 == 0:
                print('\rloading line: {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')
            words = line.strip('\r\n').split('\t')
            paperid, category, position, sentence, fieldOfStudy = words[0], words[1], int(words[2]), words[3], words[4]

            if not sentence:
                continue
            wt.write(sentence + '\n')

            for token in sentence.split(' '):
                add2dict(token, word2idx)


def get_author_reference_list(author2paper_list, paper2reference_list, paper2date):
    print('parsing user\'s reference list ...')
    author2reference_list = {}
    _cnt = 0
    _t0 = time.time()
    for author, paper_list in author2paper_list.items():
        _cnt += 1
        if _cnt % 10000 == 0:
            print('\rparsed user count: {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')
        cited_paper2cited_date = {}
        for paper in paper_list:
            if paper not in paper2date or paper not in paper2reference_list:
                continue
            date = paper2date[paper]
            reference_list = paper2reference_list[paper]
            for cited_paper in reference_list:
                if cited_paper not in paper2date:
                    continue
                if cited_paper not in cited_paper2cited_date:
                    cited_paper2cited_date[cited_paper] = date
                else:
                    if cited_paper2cited_date[cited_paper] < date:
                        cited_paper2cited_date[cited_paper] = date
        if len(cited_paper2cited_date) <= 0:
            continue
        cited_paper_info = [(key, paper2date[key], value) for key, value in cited_paper2cited_date.items()]
        cited_paper_info.sort(key=lambda x:x[1])
        author2reference_list[author] = cited_paper_info
    print()
    return author2reference_list


def output_author2reference_list(author2reference_list, filename):
    print('outputing author reference list')
    with open(filename, 'w') as wt:
        for author, ref_list in author2reference_list.items():
            paper_list = [a[0] for a in ref_list]
            paper_publich_date_list = [str(a[1]) for a in ref_list]
            paper_cited_date_list = [str(a[2]) for a in ref_list]
            wt.write('{0}\t{1}\t{2}\t{3}\n'.format(author, ','.join(paper_list), ','.join(paper_publich_date_list), ','.join(paper_cited_date_list)))


def sample_negative_and_write_to_file(outfilename, samples, neg_cnt, positive_pairs, item_list, sample_probs, remove_false_negative=False, process_id=0, process_num=4):
    with open(outfilename, 'w') as wt:
        _cnt, _total = 0, len(samples)
        _t0 = time.time()
        for sample in samples:
            _cnt += 1
            if _cnt % 1000 == 0:
                print('\rsampling process {3}:  {0} / {1}, time elapses: {2:.1f}s'.format(_cnt, _total, time.time() - _t0, process_id), end=' ')
            if _cnt % process_num != process_id:
                continue
            words = sample.split('%')
            label, user_tag, item_id = words[0].split(' ')
            wt.write(sample+'\n')
            sampled_items_indices = reparameter_sampling(neg_cnt, sample_probs)
            for sampled_item_idx in sampled_items_indices:
                sampled_item = item_list[sampled_item_idx]
                if not remove_false_negative or (words[1], sampled_item) not in positive_pairs:
                    wt.write('{0} {1} {2}%{3}\n'.format(0, user_tag, sampled_item, words[1]))
    print('\tsampling process {0} done.'.format(process_id))


def get_normalized_item_freq(item2cnt):
    keys = list(item2cnt.keys())
    values = []
    total_value = sum(item2cnt.values())
    for key in keys:
        values.append(item2cnt[key] * 1.0 / total_value)
    values = np.asarray(values, dtype=np.float32)
    return keys, values

def load_has_feature_items(InFile_paper_feature):
    item_set = set()
    with open(InFile_paper_feature, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split(' ')
            item_set.add(words[0])
    return item_set

def gen_experiment_splits(file_Author2ReferencePapers, OutFile_dir, InFile_paper_feature, tag, item_ratio=1.0, process_num=1):
    if not os.path.exists(OutFile_dir):
        os.mkdir(OutFile_dir)

    user_behavior_file = os.path.join(OutFile_dir, 'user_history_{0}.txt'.format(tag))
    train_file = os.path.join(OutFile_dir, 'train_{0}.txt'.format(tag))
    valid_file = os.path.join(OutFile_dir, 'valid_{0}.txt'.format(tag))
    test_file = os.path.join(OutFile_dir, 'test_{0}.txt'.format(tag))

    item_set = load_has_feature_items(InFile_paper_feature)
    if item_ratio < 1.0:
        _selected_items = random.sample(item_set, int(len(item_set) * item_ratio))
        item_set = set(_selected_items)

    _min_seq_len = 2
    _min_test_seq_len = 6
    _max_instance_per_user = 20
    train_neg_cnt = 4
    test_neg_cnt = 19

    train_samples, valid_samples, test_samples = [], [], []
    item2cnt = {}
    positive_pairs = set()
    print('expanding user behaviors...')
    _cnt = 0
    _t0 = time.time()
    with open(file_Author2ReferencePapers, 'r') as rd, open(user_behavior_file, 'w') as wt:
        while True:
            line = rd.readline()
            if not line:
                break
            _cnt += 1
            if _cnt % 1000 == 0:
                print('\rprocessing user number : {0}, time elapses: {1:.1f}s'.format(_cnt, time.time() - _t0), end=' ')
            words = line.strip().split('\t')
            act_items = words[1].split(',')
            act_items = [_item for _item in act_items if _item in item_set]
            act_items_len = len(act_items)
            if act_items_len <= _min_seq_len:
                continue
            for act_item in act_items:
                positive_pairs.add((words[0], act_item))

            user_behavior = ''
            for i in range(1, act_items_len):
                if i == 1:
                    user_behavior = act_items[i - 1]
                else:
                    user_behavior += ',' + act_items[i - 1]

                if act_items_len - 2 - _max_instance_per_user > i:
                    continue

                if act_items[i] not in item2cnt:
                    item2cnt[act_items[i]] = 1
                else:
                    item2cnt[act_items[i]] += 1

                user_tag = '{0}_{1}'.format(words[0], i)
                wt.write('{0} {1}\n'.format(user_tag, user_behavior))
                instance = '{0} {1} {2}%{3}'.format(1, user_tag, act_items[i], words[0])
                if act_items_len <= _min_test_seq_len:
                    train_samples.append(instance)
                else:
                    if i == act_items_len-1:
                        test_samples.append(instance)
                    elif i == act_items_len-2:
                        valid_samples.append(instance)
                    else:
                        train_samples.append(instance)
    print('done. \nsample number in train / valid / test is {0} / {1} / {2}'.format(len(train_samples), len(valid_samples), len(test_samples)))

    random.shuffle(train_samples)

    ## only keep items which have features
    item2cnt = {k: v for k, v in item2cnt.items() if k in item_set}

    item_list, sample_probs = get_normalized_item_freq(item2cnt)
    print('negative sampling for train...')
    sample_negative_and_write_to_file_wrapper(train_file, train_samples, train_neg_cnt,  positive_pairs, item_list, sample_probs, process_num=process_num)
    print('negative sampling for validation...')
    sample_negative_and_write_to_file_wrapper(valid_file, valid_samples, train_neg_cnt, positive_pairs, item_list, sample_probs, process_num=process_num)
    print('negative sampling for test...')
    sample_negative_and_write_to_file_wrapper(test_file, test_samples, test_neg_cnt, positive_pairs, item_list, sample_probs, process_num=process_num)
    print('done.')

    dump_dict_as_txt(item2cnt, os.path.join(OutFile_dir, 'item2freq.tsv'))


def sample_negative_and_write_to_file_wrapper(otuput_file, pos_samples, neg_cnt,  positive_pairs, item_list, sample_probs, process_num=1):
    p_list = []
    for i in range(process_num):
        outfile = otuput_file + '_part{0}'.format(i)
        p = Process(target=sample_negative_and_write_to_file, args=(outfile, pos_samples, neg_cnt,  positive_pairs, item_list, sample_probs, False, i, process_num))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    ### merge files and delete temporary files.
    with open(otuput_file, 'w') as wt:
        for i in range(process_num):
            infile = otuput_file + '_part{0}'.format(i)
            with open(infile, 'r') as rd:
                while True:
                    line = rd.readline()
                    if not line:
                        break
                    if len(line) > 1:
                        wt.write(line)
            os.remove(infile)

def normalize_score(pair2CocitedCnt, paper2cited_list, min_k = 10, min_score=0.1):
    res = {}
    for pair, cnt in pair2CocitedCnt.items():
        if pair[0] not in paper2cited_list or pair[1] not in paper2cited_list:
            continue
        if len(paper2cited_list[pair[0]]) < min_k or len(paper2cited_list[pair[1]]) < min_k :
            continue
        sim = math.sqrt(cnt * cnt / (4 * len(paper2cited_list[pair[0]]) * len(paper2cited_list[pair[1]])))
        if sim > min_score:
            res[pair] = sim
    return res

def gen_paper_cocitation(InFile_PaperReference, norm=True):
    paper2reference_list = load_paper_reference(InFile_PaperReference)
    paper2cited_list = reverse_dict_list(paper2reference_list)

    pair2CocitedCnt = {}
    total_cnt, cur_cnt = len(paper2cited_list), 0
    _t0 = time.time()
    for paperid, who_cite_it_list in paper2cited_list.items():
        cur_cnt += 1
        if cur_cnt % 100 == 0:
            print('\rprocess paper num {0} / {1}...time elapses: {2:.1f}s'.format(cur_cnt, total_cnt, time.time() - _t0), end='')
        for source_paperid in who_cite_it_list:
            if source_paperid not in paper2reference_list:
                continue
            for its_reference_list in paper2reference_list[source_paperid]:
                if paperid != its_reference_list:
                    pair = (paperid, its_reference_list) if paperid < its_reference_list else (its_reference_list, paperid)
                    if pair not in pair2CocitedCnt:
                        pair2CocitedCnt[pair] = 0
                    pair2CocitedCnt[pair] += 1
    print('\tDone.')


    pair2CoReferenceCnt = {}
    total_cnt, cur_cnt = len(paper2reference_list), 0
    _t0 = time.time()
    for paperid, its_reference_list in paper2reference_list.items():
        cur_cnt += 1
        if cur_cnt % 100 == 0:
            print('\rprocess paper num {0} / {1}...time elapses: {2:.1f}s'.format(cur_cnt, total_cnt, time.time() - _t0), end='')
        for reference_paperid in its_reference_list:
            if reference_paperid not in paper2cited_list:
                continue
            for its_cited_list in paper2cited_list[reference_paperid]:
                if paperid != its_cited_list:
                    pair = (paperid, its_cited_list) if paperid < its_cited_list else (its_cited_list, paperid)
                    if pair not in pair2CoReferenceCnt:
                        pair2CoReferenceCnt[pair] = 0
                    pair2CoReferenceCnt[pair] += 1
    print('\tDone.')

    if norm:
        pair2CocitedCnt = normalize_score(pair2CocitedCnt, paper2cited_list, 10, 0.145)
        pair2CoReferenceCnt = normalize_score(pair2CoReferenceCnt, paper2reference_list, 10, 0.311)

    return pair2CocitedCnt, pair2CoReferenceCnt

def year_delta_check(paper01, paper02, paper2date, threshold=365):
    if paper01 in paper2date and paper02 in paper2date:
        if math.fabs((paper2date[paper01] - paper2date[paper02]).days) <= threshold:
            return True
    return False

def author_overlap_check(paper01, paper02, paper2author_list, threshold=0.5):
    if paper01 in paper2author_list and paper02 in paper2author_list:
        n, m = len(paper2author_list[paper01]), len(paper2author_list[paper02])
        k = len(paper2author_list[paper01].intersection(paper2author_list[paper02]))
        if k/n >= threshold or k/m >= threshold:
            return True
    return False

def gen_paper_pairs_from_same_author(author2paper_list, paper2author_list, paper2date, outfile, item_set):
    total_cnt, cur_cnt = len(author2paper_list), 0
    _t0 = time.time()
    with open(outfile, 'w') as wt:
        for author, paper_list in author2paper_list.items():
            cur_cnt += 1
            if cur_cnt % 100 == 0:
                print('\rprocess author num {0} / {1}...time elapses: {2:.1f}s'.format(cur_cnt, total_cnt,
                                                                                      time.time() - _t0), end='')
            paper_list = [p for p in paper_list if p[1]==1]
            n = len(paper_list)
            if n<=1:
                continue
            for i in range(n-1):
                if paper_list[i][0] not in item_set:
                    continue
                for j in range(1, n):
                    if paper_list[j][0] not in item_set:
                        continue
                    if year_delta_check(paper_list[i][0], paper_list[j][0], paper2date) and author_overlap_check(paper_list[i][0], paper_list[j][0], paper2author_list):
                        wt.write('{0},{1}\n'.format(paper_list[i][0], paper_list[j][0]))

def gen_negative_instances(item_set, infile, outfile, neg_num):
    item_list = list(item_set)
    item_num = len(item_set)
    print('negative sampling for file {0}...'.format(os.path.basename(infile)))
    with open(infile, 'r') as rd:
        lines = rd.readlines()
    total_cnt, cur_cnt = len(lines), 0
    _t0 = time.time()
    with open(outfile, 'w') as wt:
        for line in lines:
            cur_cnt += 1
            if cur_cnt % 100 == 0:
                print('\rprocess line num {0} / {1}...time elapses: {2:.1f}s'.format(cur_cnt, total_cnt,
                                                                                       time.time() - _t0), end='')
            words = line.strip().split(',')

            wt.write('{0}\n'.format(words[0]))
            wt.write('{0}\n'.format(words[1]))

            for _ in range(neg_num):
                item = item_list[random.randint(0, item_num-1)]
                wt.write('{0}\n'.format(item))
    print('\tdone.')

def split_train_valid_file(infile_list, outdir, ratio=0.8):
    gt_pairs = set()
    for infile in infile_list:
        with open(infile, 'r') as rd:
            for line in rd:
                words = line.strip().split(',')
                pair = (words[0], words[1]) if words[0] < words[1] else (words[1], words[0])
                gt_pairs.add(pair)

    gt_pairs = list(gt_pairs)
    random.shuffle(gt_pairs)
    with open(os.path.join(outdir, 'item2item_train.txt'), 'w') as wt_train, open(os.path.join(outdir, 'item2item_valid.txt'), 'w') as wt_valid:
        for p in gt_pairs:
            if random.random() < ratio:
                wt_train.write('{0},{1}\n'.format(p[0], p[1]))
            else:
                wt_valid.write('{0},{1}\n'.format(p[0], p[1]))



#############   training word/entity embeddings
def load_np_from_txt(transE_vecfile, np_file, delimiter='\t'):
    data = []
    with open(transE_vecfile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            data.append([float(a) for a in line.strip().split(delimiter)])
    data = np.asarray(data, dtype=np.float32)
    with open(np_file, 'wb') as f:
        np.save(f, data)

def format_knowledge_embeddings(transE_vecfile, np_file):
    data = np.loadtxt(transE_vecfile, delimiter='\t')
    with open(np_file, 'wb') as f:
        np.save(f, data)

def format_word_embeddings(word_vecfile, word2id_file, np_file):
    with open(word2id_file, 'rb') as rd:
        word2id = pickle.load(rd)
    wordcnt = len(word2id)

    word_embeddings = None
    line_idx = 0
    with open(word_vecfile, 'r', encoding='utf-8') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split()
            line_idx += 1
            if line_idx == 1:
                _wordcnt, _emb_size = int(words[0]), int(words[1])
                if _wordcnt + 1 != wordcnt:  #  the 0-th word is 'NULL'
                    raise ValueError('Word number doesn\'t match in word2id ({0}) and word2embedding file ({1})!'.format(wordcnt, _wordcnt))
                word_embeddings = np.zeros(shape=(_wordcnt + 1, _emb_size), dtype=np.float32)
            else:
                _idx = word2id[words[0]]
                for i in range(1, _emb_size+1):
                    word_embeddings[_idx][i-1] = float(words[i])
    with open(np_file, 'wb') as f:
        np.save(f, word_embeddings)

def gen_context_embedding(entity_file, context_file, kg_file, dim):
    #load embedding_vec
    entity_index = 0
    entity_dict = {}
    fp_entity = open(entity_file, 'r')
    for line in fp_entity:
        linesplit = line.strip().split('\t')[:dim]
        linesplit = list(map(float, linesplit))
        entity_dict[str(entity_index)] = linesplit
        entity_index += 1
    fp_entity.close()

    #build neighbor for entity in entity_dict
    fp_kg = open(kg_file, 'r', encoding='utf-8')
    triple_num = fp_kg.readline()
    triples = fp_kg.readlines()
    kg_neighbor_dict = {}
    for triple in triples:
        linesplit = triple.strip().split(' ')
        head = linesplit[0]
        tail = linesplit[1]
        if head not in kg_neighbor_dict:
            kg_neighbor_dict[head] = set()
        kg_neighbor_dict[head].add(tail)

        if tail not in kg_neighbor_dict:
            kg_neighbor_dict[tail] = set()
        kg_neighbor_dict[tail].add(head)        
    fp_kg.close()

    context_embeddings = np.zeros([entity_index , dim])

    for entity in entity_dict:
        if entity in kg_neighbor_dict:
            context_entity = kg_neighbor_dict[entity]
            context_vecs = []
            for c_entity in context_entity:
                context_vecs.append(entity_dict[c_entity])

            context_vec = np.mean(np.asarray(context_vecs), axis=0)
            context_embeddings[int(entity)] = context_vec

    np.savetxt(context_file, context_embeddings, delimiter='\t')
    

########  data preparation for lightGCN
def load_instance_file(
        filename,
        target_triples,
        label=None
    ):
    print('load_instance_file: {0}  '.format(os.path.basename(filename)), end=' ')
    user_hist_keys = set()
    with open(filename, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split('%')
            tokens = words[0].split(' ')
            if label:
                target_triples.append((words[1], tokens[2], label))  # (userid, itemid, label)
            else:
                target_triples.append((words[1], tokens[2], tokens[0]))  #(userid, itemid, label)
            user_hist_keys.add(tokens[1])
    print('done.')
    return user_hist_keys

def write_to_file(filename, triples):
    with open(filename, 'w') as wt:
        for t in triples:
            wt.write('{0} {1} {2}\n'.format(t[0], t[1], t[2]))

def load_user_behaviors(
        user_behavior_file,
        train_triples,
        user_behavior_keys=None
    ):
    with open(user_behavior_file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split(' ')
            if user_behavior_keys and not words[0] in user_behavior_keys:
                continue
            userid = words[0].split('_')[0]
            items = words[1].split(',')
            for item in items:
                train_triples.append((userid, item, '1'))


def prepare_dataset(output_folder, input_folder, tag):
    train_triples, valid_triples = [], []

    training_user_hist_keys = load_instance_file(
        os.path.join(input_folder, 'train_{0}.txt'.format(tag)),
        train_triples
    )
    load_instance_file(
        os.path.join(input_folder, 'valid_{0}.txt'.format(tag)),
        valid_triples
    )
    load_instance_file(
        os.path.join(input_folder, 'test_{0}.txt'.format(tag)),
        valid_triples,
        label='0'
    )

    load_user_behaviors(
        os.path.join(input_folder, 'user_history_{0}.txt'.format(tag)),
        train_triples,
        training_user_hist_keys
    )

    write_to_file(os.path.join(output_folder, 'lightgcn_train_{0}.txt'.format(tag)), train_triples)
    write_to_file(os.path.join(output_folder, 'lightgcn_valid_{0}.txt'.format(tag)), valid_triples)


def group_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """
    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}
    for l, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(l)
        group_preds[k].append(p)
    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])
    return all_labels, all_preds

def load_emb_file(emb_file):
    res = {}
    with open(emb_file, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split('\t')
            values = [float(a) for a in words[1].split(' ')]
            res[words[0]] = np.asarray(values, dtype=np.float32)
    return res