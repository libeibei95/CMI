import pickle

import faiss
import pandas as pd
import torch
import os
import numpy as np
import argparse

from data_iterator_cl import *
from model_devcl import *
from tqdm import tqdm
import logging
import sys
from datetime import datetime
from time import time

sys.path.append('..')
from tensorboardX import SummaryWriter
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='wechat', help='Sess | Item')
    parser.add_argument('--dataset', type=str, default='takatak', help='wechat | takatak')
    parser.add_argument('--model', type=str, default='DevCL',
                        help='Dev | DevCL | GRU4Rec | SRGNN | STAMP | NextItNet | Caser | BERT4Rec')
    parser.add_argument('--filename', type=str, default='test', help='post filename')
    parser.add_argument('--random_seed', type=int, default=19)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout_prob', type=float, default=0)
    parser.add_argument('--n_interest', type=int, default=8)
    parser.add_argument('--n_topk_interest', type=int, default=8)  # 为每个用户最多选择 n_topk_interest个兴趣向量
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='')
    parser.add_argument('--max_iter', type=int, default=1000, help='(k)')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--test_epoch', type=int, default=1000)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--best_ckpt_path', type=str, default='runs/', help='the direction to save ckpt')
    parser.add_argument('--eval_type', type=str, default='maxscore',
                        help='avg | maxscore | his_ratios | learned_ratios | subopt | opt | gt_ratio')
    parser.add_argument('--add_cl', type=int, default=1, help='whether to add cl loss')
    parser.add_argument('--log_dir', type=str, default='log', help='the direction of log')
    parser.add_argument('--sampler', type=str, default='R', help='O|H|R: Original|Hard|Random')
    parser.add_argument('--w_sharp', type=float, default=1, help='to make item-interest distribution sharp')
    parser.add_argument('--w_orth', type=float, default=10, help=' to make system interest orth ')
    parser.add_argument('--w_uniform', type=float, default=1,
                        help='to make items uniformly distribute on global interests')
    parser.add_argument('--w_clloss', type=float, default=0.05, help='the weight of cl loss')
    parser.add_argument('--cl_type', type=str, default='emb', help='emb | prob')
    parser.add_argument('--interest_type', type=str, default='Plus',
                        help='None | Extra | Plus | ExtraPlus | Agg | ExtraAgg | ExtraAggPlus | Double | PlusDouble | PlusUid | PlusDoubleUid')
    return parser.parse_args()


def eval(model, test_data, config, phase='valid', type='avg', ks=None, ratios=None):
    '''
    Args:
        model:
        test_data:
        config:
        phase:
        type: avg|max_score|his_ratios|learned_ratios|subopt|opt|
    Returns:

    '''
    recall, ndcg = [0] * len(ks), 0
    hit = [0] * len(ks)
    num = 0
    k = config.topk
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    # try:
    gpu_index = faiss.GpuIndexFlatIP(res, model.embedding_size, flat_config)
    # print('----------')
    # with torch.no_grad():
    if config.model == 'Dev':
        item_embeds = trans_to_cpu(model.W(model.item_embedding.weight).detach()).numpy()
    elif config.model == 'Octopus':
        item_embeds = trans_to_cpu(model.item_embedding.weight.detach()).numpy()
    elif config.model == 'DevCL':
        item_embeds = model.item_embedding.weight
        item_embeds = F.normalize(item_embeds, dim=-1, p=2)
        item_embeds = trans_to_cpu(item_embeds.detach()).numpy()
    gpu_index.add(item_embeds[1:])  # filter out item zero

    model.eval()

    with torch.no_grad():
        for uids, seqs, tars, lens in tqdm(test_data):
            uids = trans_to_cuda(torch.LongTensor(uids))
            seqs = trans_to_cuda(torch.LongTensor(seqs))
            lens = trans_to_cuda(torch.LongTensor(lens))

            psnl_interest, interest_mask = model(uids, seqs, lens)
            batch_size, n_interest, embed_size = psnl_interest.shape
            user_embeds = psnl_interest.reshape(-1, psnl_interest.shape[-1])
            nrecall = int(ks[-1] * 2)
            scores, items = gpu_index.search(trans_to_cpu(user_embeds.detach()).numpy(), nrecall)
            his_seqs = trans_to_cpu(seqs).numpy()
            batch_recall, batch_hit = eval_max_score(scores.reshape(batch_size, n_interest, -1),
                                                     items.reshape(batch_size, n_interest, -1), tars, ks,
                                                     his_seqs)

            recall = [r + br for r, br in zip(recall, batch_recall)]
            hit = [h + hi for h, hi in zip(hit, batch_hit)]
            num += uids.shape[0]  # 累计样本数
    if phase == 'valid':
        if ks is None:
            logging.info('Valid: Recall@{:2d}:\t{:.4f}'.format(k, recall / num))
        else:
            for nbr_k, kk in enumerate(ks):
                logging.info('Valid: Recall@{:2d}:\t{:.4f}'.format(kk, recall[nbr_k] / num))
    else:
        if ks is None:
            logging.info('Test: Recall@{:2d}:\t{:.4f}'.format(k, recall / num))
        else:
            for nbr_k, kk in enumerate(ks):
                logging.info('Test: Recall@{:2d}:\t{:.4f}'.format(kk, recall[nbr_k] / num))
            for nbr_k, kk in enumerate(ks):
                logging.info('Test: Hit@{:2d}:\t{:.4f}'.format(kk, hit[nbr_k] / num))
    if not os.path.exists('res'):
        os.mkdir('res')
    model.train()  # reset
    if ks is None:
        return [recall / num]
    else:
        return [r / num for r in recall]


def train(writer, model, train_data, valid_data, test_data, config, ratios=None, type='train'):
    '''
    Args:
        writer:
        model:
        train_data:
        valid_data:
        test_data:
        config:
        ratios:
        type: train|finetune

    Returns:

    '''
    step = 0
    loss_sum = 0
    best_metrics = [0]
    trials = 0

    if not os.path.exists('runs'):
        os.mkdir('runs')

    best_model_path = config.best_ckpt_path

    for uids, seqs, poss, negs, lens in train_data:
        # torch.cuda.empty_cache()
        uids = trans_to_cuda(torch.LongTensor(uids))
        seqs = trans_to_cuda(torch.LongTensor(seqs))
        poss = trans_to_cuda(torch.LongTensor(poss))
        negs = trans_to_cuda(torch.LongTensor(negs))
        lens = trans_to_cuda(torch.LongTensor(lens))

        model.optimizer.zero_grad()
        step += 1
        if type == 'finetune' and step > 10 * config.test_epoch:
            break

        loss = model.calculate_loss(uids, seqs, lens, poss, negs)
        loss.backward()
        model.optimizer.step()
        loss_sum += loss.item()
        writer.add_scalar("loss", loss.item(), step)

        # record
        if step % config.test_epoch == 0:
            logging.info('Epoch:{:d}\tloss:{:4f}'.format(step // config.test_epoch, loss_sum / config.test_epoch))
            loss_sum = 0
            metrics = eval(model, valid_data, config, phase='valid', type=config.eval_type, ks=[10, 20, 50],
                           ratios=ratios)
            if metrics[-1] > best_metrics[-1]:
                if type == 'train':
                    torch.save(model.state_dict(), best_model_path)
                elif type == 'finetune':
                    torch.save(model.state_dict(), '{}_finetune'.format(best_model_path))
                best_metrics = metrics
                trials = 0
            else:
                trials += 1
                if trials >= 3:
                    model.has_unique_loss = True
                if trials > config.patience:
                    break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    config = get_args()
    SEED = config.random_seed
    setup_seed(SEED)
    data_path = '../data/{}_data'.format(config.dataset)

    train_data = TrainData(data_path, config.batch_size, maxlen=config.maxlen, sampler=config.sampler)
    valid_data = TestData(data_path, config.batch_size, maxlen=config.maxlen)
    test_data = TestData(data_path, config.batch_size, maxlen=config.maxlen, isvalid=False)
    if 'O' in config.sampler:
        config.add_cl = 0
        config.w_clloss = 0
    # print(train_data.n_user)
    config.n_item, config.n_user = train_data.n_item, train_data.n_user
    filename = '{}_{}in_{}_bs{}_s{}_cl{}_temp{}_{}_{}_{}'.format(config.dataset[:3], str(config.n_interest),
                                                           config.interest_type, config.batch_size, config.sampler, config.w_clloss, config.temp, config.cl_type,
                                                                 datetime.fromtimestamp(time()).strftime('%m%d%H%M'),
                                                                 config.filename)

    if config.filename == '':
        fileflag = input("Please input the title of the checkpoint: ")
        filename += fileflag
    config.best_ckpt_path += filename
    if not os.path.exists('runs_tensorboard'): os.mkdir('runs_tensorboard')
    writer = SummaryWriter('runs_tensorboard/{}'.format(filename))

    if not os.path.exists(config.log_dir): os.mkdir(config.log_dir)
    if os.path.exists('{}/{}'.format(config.log_dir, filename)): os.remove('{}/{}'.format(config.log_dir, filename))
    logging.basicConfig(format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        filename='{}/{}.log'.format(config.log_dir, filename),
                        level=logging.INFO)
    print(config)

    model = trans_to_cuda(DevCL(config))

    if config.eval_type == 'subopt':
        ratios = []
        n_interest = config.n_interest
        seq = [0] * 100  # 100 is larger than n_interest

        def search(rem, rem_interest):
            if rem_interest == 1:
                ratios.append(seq[:n_interest - 1] + [rem])
                return 1
            if rem == 0:
                ratios.append(seq[:n_interest - rem_interest] + [0] * rem_interest)
                return 1
            num = 0
            for i in range(0, rem + 1):
                seq[n_interest - rem_interest] = i
                num += search(rem - i, rem_interest - 1)
            return num

        nums = search(config.n_step, n_interest)
        print('There are {} kinds of allocation to choose from.'.format(nums))
    else:
        ratios = None

    train(writer, model, train_data, valid_data, test_data, config, ratios=ratios)
    model.load_state_dict(torch.load(config.best_ckpt_path))
    # eval(model, test_data, config, 'test', config.eval_type, ks=[50, 100, 200], ratios=ratios)
    eval(model, test_data, config, 'test', config.eval_type, ks=[10, 20, 50], ratios=ratios)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
    # recalculate()
    # replay()
    print('test')
    '''
    python run.py --dataset wechat --n_interest 2 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_2in_1.0uni_1.0sharp_10.0orth_211205_1800_test'
    python run.py --dataset wechat --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_4in_1.0uni_1.0sharp_10.0orth_211205_2225_test'
    python run.py --dataset wechat --n_interest 8 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_8in_1.0uni_1.0sharp_10.0orth_211205_1802_test'
    python run.py --dataset wechat --n_interest 16 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_wechat' --best_ckpt_path 'DevCL_wechat_16in_1.0uni_1.0sharp_10.0orth_211205_2034_test'

    python run.py --dataset takatak --n_interest 4 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_4in_1.0uni_1.0sharp_10.0orth_211205_2228_test'
    python run.py --dataset takatak --n_interest 8 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_8in_1.0uni_1.0sharp_10.0orth_211205_2330_test'
    python run.py --dataset takatak --n_interest 16 --w_uniform 1 --w_sharp 1 --w_orth 10  --log_dir 'log_interest_takatak' --best_ckpt_path 'DevCL_takatak_16in_1.0uni_1.0sharp_10.0orth_211205_1801_test'

    python run.py --model 'Octopus' --dataset wechat --n_interest 2 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 4 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 8 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset wechat --n_interest 16 --log_dir 'log_octopus'

    python run.py --model 'Octopus' --dataset takatak --n_interest 2 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 4 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 8 --log_dir 'log_octopus'
    python run.py --model 'Octopus' --dataset takatak --n_interest 16 --log_dir 'log_octopus'
    '''
