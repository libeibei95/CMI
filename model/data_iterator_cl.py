# coding=utf-8
import gc
import pickle
import random
import sys
import time

import numpy as np
import tqdm
from copy import deepcopy

from memory_profiler import profile
import json
import scipy.sparse as sp

class Data:
    def __init__(self, source):
        self.days = 14 if 'wechat' in source else 10
        # self.days = 14 if 'wechat' in source else 5
        self.max_day = 14 if 'wechat' in source else 29
        self.pos_seqs, self.users, self.n_user = self.read('{}/pos_seqs_renbr.txt'.format(source))
        self.n_item = max([max(se) for seq in self.pos_seqs for se in seq]) + 1

    def read(self, file):
        res, users = [], [0]
        with open(file, 'r') as f:
            for line in f:
                data = list(map(int, line.strip().split(',')))
                uid, day = data[0], data[1]
                if day < self.max_day - self.days + 1: continue
                if uid != users[-1]:
                    users.append(uid)
                    res.append([[0]] * self.days)
                res[-1][day - (self.max_day - self.days + 1)] = data[2:]
        return res, users[1:], max(users[1:]) + 1

    def get_posseq(self):
        return [[s for se in seq for s in se] for seq in self.pos_seqs]

    def get_inverted_index(self, seqs):
        '''构建倒排索引表 用于求 PMI '''
        inv_idx = {}
        for sid, seq in enumerate(seqs):
            for item in seq:
                if inv_idx.get(item) is not None:
                    inv_idx[item].add(sid)
                else:
                    inv_idx[item] = set([sid])
        return inv_idx

    def get_pmi(self, seqs):
        '''
        计算 PMI 值的函数
        Args:
            seqs:
        Returns:
        '''
        pmi = {}
        item_set = set([iid for seq in seqs for iid in seq])
        total_pairs = len(item_set) * (len(item_set) - 1) // 2
        print(total_pairs)

        for seq in tqdm.tqdm(seqs):
            for i in seq:
                for j in seq:
                    try:
                        pmi[i][j] += 1
                        pmi[i][0] += 1
                    except KeyError:
                        if pmi.get(i) is None:
                            pmi[i] = {0: 0}
                        if pmi[i].get(j) is None:
                            pmi[i][j] = 0
                        pmi[i][j] += 1
                        pmi[i][0] += 1
                    try:
                        pmi[j][i] += 1
                        pmi[j][0] += 1
                    except:
                        if pmi.get(j) is None:
                            pmi[j] = {0: 0, j: 0}
                        if pmi[j].get(i) is None:
                            pmi[j][i] = 0
                        pmi[j][i] += 1
                        pmi[j][0] += 1

        pos_pmi = {}
        cnt = 0
        for i in tqdm.tqdm(pmi):
            for j in pmi[i]:
                cnt += 1
                if j == 0: continue
                pmi[i][j] = np.log(pmi[i][j] * total_pairs / pmi[i][0] / pmi[j][0])
                if pmi[i][j] > 0:
                    if pos_pmi.get(i) is None:
                        pos_pmi[i] = {}
                    if pos_pmi.get(j) is None:
                        pos_pmi[j] = {}
                    pos_pmi[i][j] = pos_pmi[j][i] = pmi[i][j]
        print(cnt)
        return pmi, pos_pmi


class Sampler(object):
    def __init__(self, pos_seqs, max_len, min_len, n_item):
        self.pos_seqs = pos_seqs
        self.max_len = max_len
        self.min_len = min_len
        self.n_item = n_item


class OriginalSample(Sampler):
    def __init__(self, pos_seqs, max_len, min_len, n_item):
        super(OriginalSample, self).__init__(pos_seqs, max_len, min_len, n_item)

    def __call__(self, idxs):
        seqs, poss, negs = [], [], []
        for idx in idxs:
            le = len(self.pos_seqs[idx])
            tar_idx = np.random.choice(np.arange(min(self.min_len, le - 1), le))
            # tar_idx = np.random.choice(np.arange(3, le))
            seqs.append(self.pos_seqs[idx][:tar_idx][-self.max_len:])  # 截断多余的长度
            pos_list = list(self.pos_seqs[idx][tar_idx:])
            pos_set = set(pos_list)

            # tar = self.pos_seqs[idx][tar_idx]
            tar = random.choice(pos_list)
            neg = np.random.randint(1, self.n_item)
            while neg in pos_set:
                neg = np.random.randint(1, self.n_item)
            poss.append(tar)
            negs.append(neg)
        return seqs, poss, negs


class HardSplit(Sampler):
    def __init__(self, pos_seqs, max_len, min_len, pmi_path, n_item):
        super(HardSplit, self).__init__(pos_seqs, max_len, min_len, n_item)
        self.pmi_dict = pickle.load(open(pmi_path, 'rb'))
        print(sys.getsizeof(self.pmi_dict))

    def __call__(self, idxs):
        '''
        Args:
            idxs: batch_size * 1, 这里要保证每条序列的长度要大于 self.min_len*2

        Returns:

        '''
        seqs, poss, negs = [], [], []
        for idx in idxs:
            seq = self.pos_seqs[idx]
            le = len(seq)
            tar_idx = np.random.choice(np.arange(min(self.min_len * 2, le - 1), le))
            his_seq = seq[:tar_idx][- 2 * self.max_len:]  # 截断多余的长度
            # 随机选择一个种子 item, 并生成这个种子 item 的 PMI 列表，找出与之 PMI 最相近的一半 items
            seed_item = his_seq[np.random.choice(np.arange(len(his_seq)))]
            pmi_list = [self.pmi_dict[seed_item][it] if self.pmi_dict.get(seed_item) is not None and
                                                        self.pmi_dict[seed_item].get(it) is not None else 0
                        for it in his_seq]
            parti_idxs = np.argpartition(pmi_list, -len(his_seq) // 2).tolist()
            seq_one = [his_seq[i] for i in sorted(parti_idxs[: -len(his_seq) // 2])]
            seq_two = [his_seq[i] for i in sorted(parti_idxs[-len(his_seq) // 2:])]
            seqs.extend([seq_one, seq_two])

            # positive list
            pos_list = seq[tar_idx:]
            poss.extend([random.choice(pos_list), random.choice(pos_list)])

            # add negatives
            neg = np.random.randint(1, self.n_item)
            his_item_set = set(his_seq)
            for i in range(2):  # 重复两遍
                while neg in his_item_set:
                    neg = np.random.randint(1, self.n_item)
                negs.append(neg)
        return seqs, poss, negs

class RandomSplit(Sampler):
    def __init__(self, pos_seqs, max_len, min_len, n_item, overlap=False):
        super(RandomSplit, self).__init__(pos_seqs, max_len, min_len, n_item)
        self.overlap = overlap

    def __call__(self, idxs):
        '''
        Args:
            idxs: batch_size * 1, 这里要保证每条序列的长度要大于 self.min_len*2

        Returns:

        '''
        seqs, poss, negs = [], [], []
        for idx in idxs:
            seq = self.pos_seqs[idx]
            le = len(seq)
            tar_idx = np.random.choice(np.arange(min(self.min_len * 2, le - 1), le))
            his_seq = seq[:tar_idx][- 2 * self.max_len:]  # 截断多余的长度

            if len(his_seq) <= 6:
                seqs.append(his_seq[-self.max_len:])
                seqs.append(his_seq[-self.max_len:])
            elif self.overlap:
                # 允许序列出现重叠， 即重复两次，每次从序列中采样一个子序列
                for i in range(2):
                    random_floats = np.random.rand(len(his_seq))
                    chosen_idxs = [i for  i, f in enumerate(random_floats) if f >= 0.5]
                    seq_one = [his_seq[i] for idx in chosen_idxs]
                    if len(seq_one) < len(his_seq) // 2:
                        seq_one = [his_seq[i] for i in range(len(his_seq)) if i not in set(chosen_idxs)]
                    seqs.append(seq_one[-self.max_len:])
            else:
                # 将序列划分成两个互不相交的子序列
                random_idxs = random.sample(list(range(len(his_seq))), len(his_seq)//2)
                seq_one = [his_seq[idx] for idx in random_idxs]
                seq_two = [his_seq[i] for i in range(len(his_seq)) if i not in set(random_idxs)]
                seqs.extend([seq_one[-self.max_len:], seq_two[-self.max_len:]])

            # positive list
            pos_list = seq[tar_idx:]
            poss.extend([random.choice(pos_list), random.choice(pos_list)])

            # add negatives
            neg = np.random.randint(1, self.n_item)
            his_item_set = set(his_seq)
            for i in range(2):  # 重复两遍
                while neg in his_item_set:
                    neg = np.random.randint(1, self.n_item)
                negs.append(neg)
        return seqs, poss, negs



# todo: 根据 PMI 使用相似的视频替换部分视频生成新的序列
class Substitute(Sampler):
    def __init__(self, pos_seqs, max_len, min_len, topk_pmi_path, n_item):
        super(Substitute, self).__init__(pos_seqs, max_len, min_len, n_item)

        '''
        self.pmi: 假设是一个 dict, 格式为 {item1: [s1, s2, s3...]}, 其中 s* 表示与 item1 pmi 最接近的若干 item, 这里每个 item 保存最相近的 5 个。
        '''

        self.pmi_dict = pickle.load(open(topk_pmi_path, 'rb'))

    def __call__(self, idxs):
        # todo: 限制最大长度
        seqs, poss, negs = [], [], []
        for idx in idxs:
            seq = self.pos_seqs[idx]
            le = len(seq)
            tar_idx = np.random.choice(np.arange(min(self.min_len, le - 1), le))
            his_seq = seq[:tar_idx][-self.max_len:]  # 截断多余的长度
            sub_his_seq = deepcopy(his_seq)
            # 随机选择序列的一半，使用最相近的 item 替换
            sub_idxs = np.random.choice(range(len(his_seq)), len(his_seq) // 2, replace=True)
            for si in sub_idxs:
                if self.pmi_dict.get(sub_his_seq[si]) is not None:
                    sub_his_seq[si] = np.random.choice(self.pmi_dict[sub_his_seq[si]])
            seqs.extend([his_seq, sub_his_seq])

            # add positive samples
            pos_list = seq[tar_idx:]
            poss.extend([random.choice(pos_list), random.choice(pos_list)])

            # add negative samples
            neg = np.random.randint(1, self.n_item)
            his_item_set = set(his_seq)
            for i in range(2):  # 重复两遍
                while neg in his_item_set:
                    neg = np.random.randint(1, self.n_item)
                negs.append(neg)
        return seqs, poss, negs


# todo: 对于比较短的序列，寻找相似序列替换中间的某一段子序列，具体做法是寻找一头一尾的子序列进行替换。
class InsertSeq(Sampler):
    def __init__(self, pos_seqs, max_len, min_len, inv_path, n_item):
        super(InsertSeq, self).__init__(pos_seqs, max_len, min_len, n_item)
        '''
        假设这里的 inv_path 是倒排索引表，即存储一个 item 都被哪些用户交互过，格式为 {item1:}
        '''
        self.inv_tab = pickle.load(open(inv_path, 'rb'))

    def __call__(self, idxs):
        pass


class TrainData(Data):
    def __init__(self, source, batch_size, maxlen=100, minlen=3, is_finetune=False, sampler='O'):
        super(TrainData, self).__init__(source)
        self.batch_size = batch_size
        self.num_seq = len(self.pos_seqs)
        self.maxlen = maxlen
        self.minlen = minlen
        if is_finetune:
            self.pos_seqs = [[ses for ses in seq[:-1] if len(ses) > 0 and ses[0] != 0] for seq in self.pos_seqs]
        else:
            self.pos_seqs = [[ses for ses in seq[:-2] if len(ses) > 0 and ses[0] != 0] for seq in self.pos_seqs]

        self.reformat_posseqs()
        self.num_seq = len(self.pos_seqs)
        print(len(self.pos_seqs))
        if 'O' in sampler:
            self.sampler = OriginalSample(self.pos_seqs, self.maxlen, self.minlen, self.n_item)
        elif 'R' in sampler:
            self.sampler = RandomSplit(self.pos_seqs, self.maxlen, self.minlen, self.n_item)
        elif 'H' in sampler:
            self.sampler = HardSplit(self.pos_seqs, self.maxlen, self.minlen, '{}/pospmi.pkl'.format(source), self.n_item)
        print('finish init sampler')

    def reformat_posseqs(self):
        self.pos_seqs = [[s for se in seq for s in se] for seq in self.pos_seqs]
        self.pos_seqs = [seq for seq in self.pos_seqs if len(seq) >= 4]  # 约束长度大于等于 3

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        # return self.next_item_batch() # to test next-click item style
        idxs = np.random.choice(np.arange(0, self.num_seq), size=self.batch_size, replace=False)
        uids = list(map(lambda idx: self.users[idx], idxs))
        seqs, poss, negs = self.sampler(idxs)
        # for idx in idxs:
        #     le = len(self.pos_seqs[idx])
        #     tar_idx = np.random.choice(np.arange(3, le))
        #     # tar_idx = np.random.choice(np.arange(3, le))
        #     seqs.append(self.pos_seqs[idx][:tar_idx])
        #     pos_list = list(self.pos_seqs[idx][tar_idx:])
        #     pos_set = set(pos_list)
        #
        #     # tar = self.pos_seqs[idx][tar_idx]
        #     tar = random.choice(pos_list)
        #     neg = np.random.randint(1, self.n_item)
        #     while neg in pos_set:
        #         neg = np.random.randint(1, self.n_item)
        #     poss.append(tar)
        #     negs.append(neg)

        lens = [len(seq) for seq in seqs]
        max_seqlen = min(max(lens), self.maxlen)
        seqs = [seq + [0] * (max_seqlen - len(seq)) if len(seq) <= max_seqlen else seq[len(seq) - max_seqlen:] for seq
                in seqs]
        lens = [min(le, max_seqlen) for le in lens]
        if 0 in set(lens):
            for i, j in enumerate(lens):
                if j==0:
                    print(i)

        return uids, seqs, poss, negs, lens


class TestData(Data):
    def __init__(self, source, batch_size, isvalid=True, maxlen=100):
        super(TestData, self).__init__(source)
        self.inv_idx = self.get_inverted_index(self.get_posseq())
        self.batch_size = batch_size
        self.num_seq = len(self.pos_seqs)
        self.curr_batch = 0
        self.taridx = -2 if isvalid else -1
        self.maxlen = maxlen

        # 筛选 taridx 有交互的用户，并把非空序列拼接在一块儿。
        self.pos_seqs = [[ses for ses in seq[:self.taridx] if len(ses) > 0 and ses[0] != 0] + [seq[self.taridx]] for seq
                         in self.pos_seqs
                         if len(seq[self.taridx]) > 0 and seq[self.taridx][0] != 0]
        self.pos_seqs = [seq for seq in self.pos_seqs if len(seq) > 1]
        self.num_seq = len(self.pos_seqs)
        self.nbatch = self.num_seq // self.batch_size
        if self.num_seq % self.batch_size:
            self.nbatch += 1
        print(len(self.pos_seqs))

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        idxs = np.arange(self.curr_batch * self.batch_size, min(self.num_seq, (self.curr_batch + 1) * self.batch_size))
        uids = list(map(lambda idx: self.users[idx], idxs))
        if len(idxs) == 0:
            return
        seqs, tars = [], []
        for idx in idxs:
            his_seq = [item for ses in self.pos_seqs[idx][:-1] for item in ses]
            gts = self.pos_seqs[idx][-1]
            seqs.append(his_seq)
            tars.append(gts)

        lens = list(map(len, seqs))
        max_seqlen = min(max(lens), self.maxlen)
        seqs = [seq + [0] * (max_seqlen - len(seq)) if len(seq) <= max_seqlen else seq[len(seq) - max_seqlen:] for seq
                in seqs]
        lens = [min(le, max_seqlen) for le in lens]
        self.curr_batch += 1
        if self.curr_batch >= self.nbatch:
            self.curr_batch = 0
            raise StopIteration
        return uids, seqs, tars, lens

# @profile
def seri_dok():
    # arr = pickle.load(open('pmi_coo.pkl', 'rb'))
    # arr = arr.todok()
    # pickle.dump(arr, open('pmi_dok.pkl', 'wb'))
    # print(arr[(0, 0)])
    a = pickle.load(open('../data/wechat_data/pospmi.pkl', 'rb'))

    n_item = max(a.keys())
    print(n_item)
    arr = sp.dok_matrix((n_item+1, n_item+1), dtype=np.float32)
    for k in tqdm.tqdm(a.keys()):
        for kk in a[k]:
            arr[k, kk] = a[k][kk]
    print('finish rows cols vals')
    pickle.dump(arr, open('pmi_dok.pkl', 'wb'))

def reseri_dok():
    arr = pickle.load(open('../data/wechat_data/pmi_dok.pkl', 'rb'))
    print('finish read pmi_dok')
    print(arr[(0, 0)])
    pickle.dump(arr, open('pmi_dok.pkl', 'wb'))
    # 序列化成 dok, 发现最后卡在了存储这一步，也放弃了，


def seri_json():
    arr = pickle.load(open('../data/wechat_data/pospmi.pkl', 'rb'))
    arr_str = json.dumps(arr)
    with open('pmi_json.txt', 'w') as f:
        f.write(arr_str)
@profile
def reseri_json():
    t1 = time.time()
    f = open('pmi_json.txt', 'r')
    arr_str = json.loads(f.readline())
    gc.collect()
    t3 = time.time()
    print(sys.getsizeof(arr_str))
    print(t3-t1)
    #序列化 json 后在加载的时候约需要 10 G 内存，比 pkl 好一点儿，但还是远远大于数据本身的大小，放弃了

if __name__ == '__main__':
    train_data = TrainData('../data/takatak_data', 512)
    for tr in train_data:
        len(tr)
    # reseri_json()
    # a = dict([(p, dict([(i, j) for i, j in enumerate(range(10))])) for p in range(2)])
    # a_str = json.dumps(a)
    # b = json.loads(a_str)
    # print(a)
    # print(a_str)
    # print(b)
    # test()
    # todo: 尝试保存为文本文件或者 json 文件
    # train_data = TrainData('../data/wechat_data', 16)
    # for tr in train_data:
    #     print(len(tr))
    # test_data = TestData('../data/wechat_data', 16, isvalid=False)
    # seqs = [[item for ses in seq[:-1] for item in ses] for seq in test_data.pos_seqs]
    # pmi, pospmi = test_data.get_pmi(seqs)
    # pickle.dump(pospmi, open('wechat_pospmi.pkl', 'wb'))
    # pickle.dump(pmi, open('wechat_pmi.pkl', 'wb'))
