import os
import random

data_root_path = '../../MINDsmall/'

print("building MINDsmall train, valid, user_history")

#download data first
import zipfile
def unzip_file(zip_src, dst_dir):
    fz = zipfile.ZipFile(zip_src, 'r')
    for file in fz.namelist():
        fz.extract(file, dst_dir)

from urllib.request import urlretrieve
url_train = "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
urlretrieve(url_train, data_root_path+"train/MINDsmall_train.zip")
unzip_file(data_root_path+"train/MINDsmall_train.zip", data_root_path+"train/")
os.remove(data_root_path+"train/MINDsmall_train.zip")
url_valid = "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
urlretrieve(url_valid, data_root_path+"dev/MINDsmall_dev.zip")
unzip_file(data_root_path+"dev/MINDsmall_dev.zip", data_root_path+"dev/")
os.remove(data_root_path+"dev/MINDsmall_dev.zip")


def read_clickhistory(path, filename):
    userid_history = {}
    with open(os.path.join(path, filename)) as f:
        lines = f.readlines()
    sessions = []
    for i in range(len(lines)):
        _, userid, imp_time, click, imps = lines[i].strip().split('\t')
        clikcs = click.split(' ')
        pos = []
        neg = []
        imps = imps.split(' ')
        for imp in imps:
            if imp.split('-')[1] == "1":
                pos.append(imp.split('-')[0])
            else:
                neg.append(imp.split('-')[0])
        userid_history[userid] = clikcs
        sessions.append([userid, clikcs, pos, neg])
    return sessions, userid_history


train_session, train_history = read_clickhistory(data_root_path+"train/",'behaviors.tsv')
valid_session, valid_history = read_clickhistory(data_root_path+"dev/",'behaviors.tsv')


def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)

npratio=4

def get_train_input(session, train_file_path):
    fp_train = open(train_file_path, 'w', encoding='utf-8')
    for sess_id in range(len(session)):
        sess = session[sess_id]
        userid, _, poss, negs = sess
        for i in range(len(poss)):
            pos = poss[i]
            neg = newsample(negs, npratio)
            fp_train.write("1 "+"train_"+userid+" "+pos+'\n')
            for neg_ins in neg:
                fp_train.write("0 " + "train_" + userid + " " + neg_ins + '\n')
    fp_train.close()

def get_valid_input(session, valid_file_path):
    fp_valid = open(valid_file_path, 'w', encoding='utf-8')
    for sess_id in range(len(session)):
        userid, _, poss, negs = session[sess_id]
        for i in range(len(poss)):
            fp_valid.write("1 " + "valid_" + userid + " " + poss[i] +"%"+ str(sess_id) +'\n')
        for i in range(len(negs)):
            fp_valid.write("0 " + "valid_" + userid + " " + negs[i] +"%"+ str(sess_id) +'\n')
    fp_valid.close()



def get_user_history(train_history, valid_history, user_history_path):
    fp_user_history = open(user_history_path, 'w', encoding='utf-8')
    for userid in train_history:
        fp_user_history.write("train_"+userid+' '+','.join(train_history[userid])+'\n')
    for userid in valid_history:
        fp_user_history.write("valid_"+userid+' '+','.join(valid_history[userid])+'\n')
    fp_user_history.close()

get_train_input(train_session, data_root_path + "train_mind_small.txt")
get_valid_input(valid_session, data_root_path + "valid_mind_small.txt")

get_user_history(train_history, valid_history, data_root_path+"user_history_small.txt")

print("finished building MINDsmall train, valid, user_history")