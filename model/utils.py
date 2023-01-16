# --coding=utf-8---
import torch
import numpy as np
from functools import reduce, partial
from multiprocessing import Pool
from collections import Counter


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# ndcg
def get_ndcg(y_pred, y_true):
    '''
    :param y_pred: a list [1，3], 预测为正样本的 item 编号，且按照预测概率排序
    :param y_true: a list or set of ground truth items [1, 2, 3, 4, ...],
    :return:
    '''
    y_true = set(y_true)
    dcg = np.sum([1 / np.log2(idx + 2) for idx, y in enumerate(y_pred) if y in y_true])
    idcg = np.sum(1 / np.log2(np.arange(len(y_true)) + 2))
    return dcg / idcg


''' AUC 需要在所有 item 上的评分，并且需要排序，暂时先不使用'''


def get_auc(y_score, tars):
    '''

    :param y_score: a list of scores with length nitem
    :param tars: a list of tars
    :return:
    '''
    f = [[y, idx + 1] for idx, y in enumerate(y_score)]
    rank = [l for s, l in sorted(f, key=lambda x: x[0])]
    tars = set(tars)
    ranklist = [i + 1 for i, r in enumerate(rank) if r in tars]
    pos_num = len(tars)
    neg_num = len(rank) - pos_num
    auc = (sum(ranklist) - pos_num * (pos_num + 1) / 2) / (pos_num * neg_num)
    return auc


def get_gauc(y_scores, targets):
    gauc = 0
    for ys, yl in zip(y_scores, targets):
        gauc += get_auc(ys, yl)
    return gauc / len(y_scores)


def eval_max_score(scores, items, tars, ks, his_seqs, full_base_scores=None):
    # def eval_max_score(scores, tars, ks):
    '''
    recall according to max score rule
    Args:
        scores: batch_size * n_interest * ks[-1]
        tars: batch_size, each element is a list containing ground-truth items
        k: top k
    Returns:
    '''
    items = list(map(lambda x: x + 1, items))  # 编号+1
    batch_recall = [0] * len(ks)
    batch_hit = [0] * len(ks)

    for sc, it, tar, seq in zip(scores, items, tars, his_seqs):
        item_list = list(zip(it.reshape(-1), sc.reshape(-1)))
        item_list.sort(key=lambda x: x[1], reverse=True)
        item_list = list(map(lambda x: x[0], item_list))
        his_item_set = set(list(seq))
        for nbr_k, k in enumerate(ks):
            rec_its = set(list(item_list[:k])) - his_item_set
            tmp_idx = k
            while len(rec_its) < k and tmp_idx < len(item_list):
                if item_list[tmp_idx] not in his_item_set:
                    rec_its.add(item_list[tmp_idx])
                tmp_idx += 1
            rec_its = rec_its.intersection(set(tar))
            batch_recall[nbr_k] += len(rec_its) / len(tar)
            if len(rec_its):
                batch_hit[nbr_k] += 1
    # '''
    return batch_recall, batch_hit

def pmi_helper(seq, inv_idx, pmi_mem, npair):
    pmi = []
    for i, item1 in enumerate(seq):
        for j, item2 in enumerate(seq):
            if i != j:
                if pmi_mem.get(item1) is None:
                    pmi_mem[item1] = {}
                if pmi_mem.get(item2) is None:
                    pmi_mem[item2] = {}
                if pmi_mem[item1].get(item2) is None:
                    pmi_mem[item1][item2] = np.log2(
                        1 + len(inv_idx[item1].intersection(inv_idx[item2])) * npair / (len(inv_idx[item1])) / (
                            len(inv_idx[item2])))
                if pmi_mem[item2].get(item1) is None:
                    pmi_mem[item2][item1] = pmi_mem[item1][item2]
                pmi.append(pmi_mem[item1][item2])
    avg_pmi = sum(pmi) / len(pmi)
    pmi_var = np.var(avg_pmi)
    max_pmi = max(pmi)
    min_pmi = min(pmi)
    # todo: here to return pmi_var, max_pmi, min_pmi
    return avg_pmi


def eval_max_score_pmi(scores, items, inv_idx, ks, pmi_mem, his_seqs, npair, full_base_scores=None):
    total_pmi, max_pmi, min_pmi = [0] * len(ks), [0] * len(ks), [9999999] * len(ks)
    batch_pmi = [[]] * len(ks)
    cnt = 0
    items = list(map(lambda x: x + 1, items))  # 编号+1
    for sc, it, seq in zip(scores, items, his_seqs):
        his_item_set = set(list(seq))
        # item_list = list(zip(it.reshape(-1), (sc * mask).reshape(-1)))
        item_list = list(zip(it.reshape(-1), sc.reshape(-1)))
        item_list.sort(key=lambda x: x[1], reverse=True)
        item_list = list(map(lambda x: x[0], item_list))
        for nbr_k, k in enumerate(ks):
            rec_its = set(list(item_list[:k])) - his_item_set
            tmp_idx = k
            while len(rec_its) < k and tmp_idx < len(item_list):
                if item_list[tmp_idx] not in his_item_set:
                    rec_its.add(item_list[tmp_idx])
                tmp_idx += 1
            assert len(rec_its) == k
            tmp_pmi = pmi_helper(list(rec_its), inv_idx, pmi_mem, npair)
            batch_pmi[nbr_k].append(tmp_pmi)
            total_pmi[nbr_k] += tmp_pmi
            max_pmi[nbr_k] = max(max_pmi[nbr_k], tmp_pmi)
            min_pmi[nbr_k] = min(min_pmi[nbr_k], tmp_pmi)
    return total_pmi, max_pmi, min_pmi


def eval_his_ratio(scores, tars, ks, his_ratios):
    '''
        recall according to max score rule
        Args:
            scores: batch_size * n_interest * n_item
            tars: batch_size, each element is a list containing ground-truth items
            k: top k
            his_ratios: batch_size * n_interest
        Returns:
    '''
    batch_size, n_interest, _ = scores.shape
    sub, sub_idx = torch.topk(scores, ks[-1], dim=-1)  # sub_idx: batch_size * n_interest * k
    sub_idx = trans_to_cpu(sub_idx).detach().numpy()

    batch_recall = [0] * len(ks)
    for i, (idx, tar) in enumerate(zip(sub_idx, tars)):
        for nbr_k, k in enumerate(ks):
            recall_nums = [int(his_ratios[i][j] * k) for j in range(len(his_ratios[i]) - 1)]
            recall_nums.append(k - sum(recall_nums))
            recall_items = [set(idx[nbr_r][:r]) for nbr_r, r in enumerate(recall_nums)]
            recall_items = reduce(lambda x, y: x.union(y), recall_items)
            batch_recall[nbr_k] += len(recall_items.intersection(set(tar))) / len(tar)
    return batch_recall


def find_best_recall(n_step, ratios, ks, data):
    '''
    Args:
        each_sub_idx: n_interest * topk
        tar: n_tar
    Returns:
    '''
    sub_idx, tar = data
    tar = set(list(tar))
    n_interest = sub_idx.shape[0]
    # 寻找最佳划分比例
    k = ks[-1]  # 取决于以哪个指标为准，这里以 Recall@200, 如果以Recall
    step = k // n_step
    hit_item_mem = []
    for each_sub_idx in sub_idx:
        hit_items = []
        for idx, r in enumerate(range(n_step)):
            if idx == 0:
                hit_items.append(set(each_sub_idx[:step]).intersection(tar))
            else:
                hit_items.append(
                    hit_items[-1].union(set(each_sub_idx[r * step:(r + 1) * step]).intersection(tar)))
        hit_item_mem.append(hit_items)

    max_recall = 0
    max_recall_ratio = []

    # todo: calculate recall for different k
    for each_ratio in ratios:
        recall_sets = [hit_item_mem[idx][r - 1] for idx, r in enumerate(each_ratio) if r > 0]
        recall_rate = len(reduce(lambda x, y: x.union(y), recall_sets)) / len(tar)
        if recall_rate >= max_recall:
            max_recall = recall_rate
            max_recall_ratio = each_ratio

    # todo: calculate recalls @ different ks
    recall_res = [0] * len(ks)
    for nbr_k, k in enumerate(ks):
        step = k // n_step
        recs = [set(list(sub_idx[i][:max_recall_ratio[i] * step])) for i in range(n_interest)]
        recs = reduce(lambda x, y: x.union(y), recs)
        recall_res[nbr_k] = len(recs.intersection(tar)) / len(tar)
    return recall_res


def eval_avg_ratio(scores, tars, ks):
    '''
    allocate recall the same ratios
    Args:
        scores: n_interest * n_item
        tars: batch_size, each element is a list containing ground-truth items
        k: top k
    Returns:
    '''

    batch_size, n_interest, _ = scores.shape
    sub, sub_idx = torch.topk(scores, ks[-1] // n_interest, dim=-1)  # sub_idx: batch_size * n_interest * k
    sub_idx = trans_to_cpu(sub_idx).detach().numpy()
    batch_recall = [0] * len(ks)
    for idx, tar in zip(sub_idx, tars):
        tar = set(tar)
        for nbr_k, k in enumerate(ks):
            idx_list = [set(list(each_idx[:(k // n_interest)])) for each_idx in idx]
            rec = reduce(lambda x, y: x.union(y), idx_list)
            batch_recall[nbr_k] += len(rec.intersection(tar)) / len(tar)
    return batch_recall


def eval_subopt(scores, tars, ks, ratios, n_step):
    '''

    Args:
        scores:
        tars:
        ks:
        ratios:
        min_ratio: 最小比例，例如如果共召回200个，每10个一组，那么一组的比例就是 1/20, 共召回100个，每5个一组，共召回 50 个的时候，就不能整除了，就得按 1/25 了。

    Returns:

    '''
    ''' allocate recall according the subopt ratios, for example, 10 for a group'''
    batch_size, n_interest, _ = scores.shape
    sub, sub_idx = torch.topk(scores, ks[-1], dim=-1)  # sub_idx: batch_size * n_interest * k
    sub_idx = trans_to_cpu(sub_idx).detach().numpy()
    for k in ks:
        if k % n_step != 0:
            raise ValueError("Invalid step number")
    with Pool(16) as p:
        res = p.map(partial(find_best_recall, n_step, ratios, ks), list(zip(sub_idx, tars)))
    final_recall = np.sum(np.array(res), axis=0)
    return final_recall


def eval_gt_ratio(scores, tars, ks):
    batch_size, n_interest, n_item = scores.shape
    interest_idxs = trans_to_cpu(torch.argmax(scores, dim=1)).detach().numpy()  # batch_size * n_item
    sub, sub_idx = torch.topk(scores, ks[-1], dim=-1)  # sub_idx: batch_size * n_interest * k
    sub_idx = trans_to_cpu(sub_idx).detach().numpy()

    # todo: recall for each interest embedding with different ratios and check if it is better than the origin gru4rec.
    batch_recall = [0] * len(ks)
    for i, (idx, tar) in enumerate(zip(sub_idx, tars)):
        tar_inidx = interest_idxs[i][tar]
        tar = set(tar)
        inte_ratio_dict = Counter(tar_inidx)
        inte_ratio_dict = {k: inte_ratio_dict[k] / len(tar) for k in inte_ratio_dict}
        recall_nums = [int(inte_ratio_dict.get(r, 0) * 200) for r in range(n_interest - 1)]
        recall_nums.append(ks[-1] - sum(recall_nums))
        pred_res = []
        for i, n in enumerate(recall_nums):
            pred_res += list(idx[i][:n])
        for nbr_k, k in enumerate(ks):
            batch_recall[nbr_k] += len(set(pred_res[:k]).intersection(tar)) / len(tar)
    return batch_recall


def eval_learned_ratio(sub_idx, tars, ks, model, feature):
    ''' allocate recall according to the trained model'''
    # todo:
    return "Not Implemented"


def eval_opt(sub_idx, tars, k):
    '''calculate the optimized recall, it's time-expensive, be careful to use'''
    # todo:
    return "Not Implemented"
