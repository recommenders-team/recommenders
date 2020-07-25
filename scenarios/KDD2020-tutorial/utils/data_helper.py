from datetime import datetime
import codecs
import os

def load_paper_reference(infile):
    print('loading {0}...'.format(os.path.basename(infile)))
    paper2reference_list = {}
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split('\t')
            if len(words) < 2:
                continue
            if words[0] not in paper2reference_list:
                paper2reference_list[words[0]] = []
            paper2reference_list[words[0]].append(words[1])
    return paper2reference_list

def load_paper_date(infile):
    print('loading {0}...'.format(os.path.basename(infile)))
    paper2date = {}
    with open(infile, 'r', encoding='utf-8', newline='\r\n') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split('\t')
            if words[8]:
                paper2date[words[0]] = datetime.strptime(words[8],  '%m/%d/%Y %I:%M:%S %p')
            else:
                paper2date[words[0]] = datetime.strptime('1/1/1970 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    return paper2date

def load_paper_author(infile):
    print('loading {0}...'.format(os.path.basename(infile)))
    paper2author = {}
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split('\t')
            paper2author[words[0]] = words[1]
    return paper2author

def load_author_paperlist(infile):
    print('loading {0}...'.format(os.path.basename(infile)))
    author2paper_list = {}
    with open(infile, 'r', newline='\r\n', encoding='utf-8') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip('\r\n').split('\t')
            if words[1] not in author2paper_list:
                author2paper_list[words[1]] = []
            author2paper_list[words[1]].append(words[0])
    return author2paper_list

def load_paper_author_relation(infile):
    print('loading {0}...'.format(os.path.basename(infile)))
    author2paper_list = {}
    paper2author_set = {}
    with open(infile, 'r', newline='\r\n', encoding='utf-8') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip('\r\n').split('\t')
            order = int(words[3])
            if words[1] not in author2paper_list:
                author2paper_list[words[1]] = []
            author2paper_list[words[1]].append((words[0], order))
            if words[0] not in paper2author_set:
                paper2author_set[words[0]] = set()
                paper2author_set[words[0]].add(words[1])
    return author2paper_list, paper2author_set

