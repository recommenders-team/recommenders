import os
from nltk.tokenize import RegexpTokenizer
import numpy as np
import json

data_root_path = '../../MINDsmall/'

MAX_SENTENCE = 10
word_embedding_dim = 100

tokenizer = RegexpTokenizer(r'\w+')

news_words={}
news_entities={}

print("building MINDsmall doc_features")

def read_news(path,filenames):
    with open(os.path.join(path,filenames), encoding='utf-8') as f:
        lines=f.readlines()
    for line in lines:
        splited = line.strip('\n').split('\t')
        news_words[splited[0]] = tokenizer.tokenize(splited[3].lower())
        news_entities[splited[0]] = []
        for entity in json.loads(splited[6]):
            news_entities[splited[0]].append((entity['SurfaceForms'], entity['WikidataId']))

read_news(data_root_path, "train/news.tsv")
read_news(data_root_path, "dev/news.tsv")


#download pretrain word vec
import zipfile
def unzip_file(zip_src, dst_dir):
    fz = zipfile.ZipFile(zip_src, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)


from urllib.request import urlretrieve
url = "http://nlp.stanford.edu/data/glove.6B.zip"
urlretrieve(url, data_root_path+"glove.6B.zip")
unzip_file(data_root_path+"glove.6B.zip", data_root_path)
os.remove(data_root_path+"glove.6B.zip")

word_set = set()
word_embedding_dict = {}
fp_pretrain_vec = open(data_root_path+"glove.6B."+str(word_embedding_dim)+"d.txt", 'r', encoding='utf-8')
for line in fp_pretrain_vec:
    linesplit = line.split(' ')
    word_set.add(linesplit[0])
    word_embedding_dict[linesplit[0]] = np.asarray(list(map(float,linesplit[1:])))
fp_pretrain_vec.close()

entity_embedding_dict = {}
fp_entity_vec_train = open(data_root_path+"train/entity_embedding.vec", 'r', encoding='utf-8')
for line in fp_entity_vec_train:
    linesplit = line.split()
    entity_embedding_dict[linesplit[0]] = np.asarray(list(map(float,linesplit[1:])))
fp_entity_vec_train.close()
fp_entity_vec_valid = open(data_root_path+"dev/entity_embedding.vec", 'r', encoding='utf-8')
for line in fp_entity_vec_valid:
    linesplit = line.split()
    entity_embedding_dict[linesplit[0]] = np.asarray(list(map(float,linesplit[1:])))
fp_entity_vec_valid.close()

word_dict = {}
word_index = 1

news_word_string_dict = {}
news_entity_string_dict = {}

entity2index = {}
entity_index = 1

for doc_id in news_words:
    news_word_string_dict[doc_id] = [0 for n in range(MAX_SENTENCE)]
    news_entity_string_dict[doc_id] = [0 for n in range(MAX_SENTENCE)]
    surfaceform_entityids = news_entities[doc_id]
    for item in surfaceform_entityids:
        if item[1] not in entity2index and item[1] in entity_embedding_dict:
            entity2index[item[1]] = entity_index
            entity_index = entity_index + 1
    for i in range(len(news_words[doc_id])):
        if news_words[doc_id][i] in word_embedding_dict:
            if news_words[doc_id][i] not in word_dict:
                word_dict[news_words[doc_id][i]] = word_index
                word_index = word_index + 1
                news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
            else:
                news_word_string_dict[doc_id][i] = word_dict[news_words[doc_id][i]]
            for item in surfaceform_entityids:
                for surface in item[0]:
                    for surface_word in surface.split(' '):
                        if news_words[doc_id][i] == surface_word.lower():
                            if item[1] in entity_embedding_dict:
                                news_entity_string_dict[doc_id][i] = entity2index[item[1]]
        if i == MAX_SENTENCE-1:
            break

word_embeddings = np.zeros([word_index, word_embedding_dim])
for word in word_dict:
    word_embeddings[word_dict[word]] = word_embedding_dict[word]

entity_embeddings = np.zeros([entity_index, 100]) # entity_embedding_dim
for entity in entity2index:
    entity_embeddings[entity2index[entity]] = entity_embedding_dict[entity]


fp_doc_string = open(data_root_path+"doc_feature.txt", 'w', encoding='utf-8')
for doc_id in news_word_string_dict:
    fp_doc_string.write(doc_id+' '+','.join(list(map(str,news_word_string_dict[doc_id])))+' '+','.join(list(map(str,news_entity_string_dict[doc_id])))+'\n')

np.save((data_root_path+'word_embeddings_5w_' + str(word_embedding_dim)), word_embeddings)
np.save((data_root_path+'entity_embeddings_5w_' + str(100)), entity_embeddings)

print("finished building MINDsmall doc_features")
