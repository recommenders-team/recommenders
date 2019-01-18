
import numpy as np
import os
import random

class DataGenerator(object):

    def __init__(self, field_num=50, feature_num=10000, dim=20, mean=0, sigma=1.0):

        # check validity
        if feature_num < field_num * 2:
            feature_num = field_num * 2

        self.field_num = field_num
        self.feature_num = feature_num
        self.dim = dim
        self.mean = mean
        self.sigma = sigma

        self.embeddings = None
        self.feature2field = None
        self.field2featurelist = None
        self.patterns = None


    def gen_embeddings(self):
        # embedding size:  (feature_num , dim)
        self.embeddings = np.random.normal(self.mean, self.sigma, (self.feature_num, self.dim))

        # assign each feature to one field
        self.feature2field = {}
        self.field2featurelist = {}

        field2featurecnt = np.random.multinomial(self.feature_num - self.field_num*2, np.random.dirichlet(np.ones(self.field_num), size=1)[0], size=1)[0] + 2

        cur_idx = 0
        for i in range(self.field_num):
            end_idx = cur_idx + field2featurecnt[i]

            self.field2featurelist[i] = list(range(cur_idx, end_idx))
            for j in range(cur_idx, end_idx):
                self.feature2field[j] = i

            cur_idx = end_idx


    def gen_patterns(self, max_pattern_num=10, max_order=2, skew=1):
        # patterns:  list of ([field_idx], weight)
        existing_patterns = set()
        retry_times = 100
        self.patterns = []
        while True:
            if retry_times<0 or len(self.patterns)>=max_pattern_num:
                break
            cur_order = np.random.randint(2,max_order+1)
            cur_pattern = np.sort(np.random.choice(self.field_num, cur_order, replace=False))
            key = '_'.join(map(str,cur_pattern))
            if key in existing_patterns:
                retry_times -= 1
                continue

            #self.patterns.append((cur_pattern, np.random.uniform(0,1)))
            self.patterns.append((cur_pattern, np.random.beta(2, cur_order*skew)))



    def write_embeddings_to_file(self, outfile):
        np.save(outfile, self.embeddings)

    def write_patterns_to_file(self, outfile):
        with open(outfile, 'w') as wt:
            for p in self.patterns:
                wt.write('{0:s}\t{1:f}\n'.format(','.join(map(str,p[0])),p[1]))

    def write_field2featurelist_to_file(self, outfile):
        with open(outfile, 'w') as wt:
            for k,v in self.field2featurelist.items():
                wt.write('{0:d}\t{1:s}\n'.format(k, ','.join(map(str,v))))

    def load_feature_field_mapping_from_file(self, infile):
        self.field2featurelist = {}
        self.feature2field = {}
        with open(infile, 'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line.strip().split('\t')
                cur_feautres = [int(t) for t in words[1].split(',')]
                cur_field = int(words[0])
                self.field2featurelist[cur_field] = cur_feautres
                for one_feature in cur_feautres:
                    self.feature2field[one_feature] = cur_field


    def load_embeddings_from_file(self, infile):
        self.embeddings = np.load(infile)

    def load_patterns_from_file(self, infile):
        self.patterns = []
        with open(infile, 'r') as rd:
            while True:
                line = rd.readline()
                if not line:
                    break
                words = line.strip().split('\t')
                key = [int(t) for t in words[0].split(',')]
                val = float(words[1])
                self.patterns.append((key,val))

    def gen_instances_to_file(self, k, outfile):
        wt = open(outfile, 'w')
        for _ in range(k):
            instances_features = []
            for i in range(self.field_num):
                instances_features.append(np.random.choice(self.field2featurelist[i], 1, replace=False)[0])

            score = 0
            for p in self.patterns:
                cur_vec = np.ones(self.dim)
                for pf in p[0]:
                    cur_vec = np.multiply(cur_vec, self.embeddings[instances_features[pf]])
                score += np.sum(cur_vec)*p[1]

            if abs(score) < 1:
                continue
            wt.write('{0:.3f},{1:s}\n'.format(1 if score > 0 else 0, ','.join([str(t) for t in instances_features])))

        wt.close()




def convert2FFMformat(infile, outfile):
    outdirname = os.path.dirname(outfile)
    os.makedirs(outdirname, exist_ok=True)
    with open(infile,'r') as rd, open(outfile, 'w') as wt:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line.strip().split(',')
            for i in range(1, len(words)):
                words[i] = '{0}:{1}:1'.format(i, 1+int(words[i]))
            wt.write(' '.join(words)+'\n')

def split_file(infile, outdir, split_ratios):
    os.makedirs(outdir, exist_ok=True)
    outfiles = []
    for i in range(len(split_ratios)):
        outfiles.append(open(os.path.join(outdir, 'part_'+str(i)), 'w'))


    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            r = random.random()
            for i in range(len(split_ratios)):
                if split_ratios[i]>=r:
                    outfiles[i].write(line)
                    break

    for wt in outfiles:
        wt.close()
