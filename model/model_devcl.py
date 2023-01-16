import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
import torch.nn.functional as F
from loss import *
from utils import *
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class AttLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttLayer, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, qvec, kmat):
        '''
        Args:
            qvec: batch_size * hidden_dim
            kmat: batch_size * n_interest * hidden_dim

        Returns:
        '''
        # scores = self.linear(qvec).unsqueeze(1).matmul(kmat.transpose(1, 2)) # bs*ni
        scores = qvec.unsqueeze(1).matmul(kmat.transpose(1, 2)) / 0.1  # bs*ni
        probs = torch.softmax(scores, dim=-1)
        interest = torch.matmul(probs, kmat).squeeze()  # bs*embed_size
        return interest


class DevCL(nn.Module):

    def __init__(self, config):
        super(DevCL, self).__init__()

        # load parameters info
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_prob = config.dropout_prob
        self.n_items = config.n_item
        self.n_users = config.n_user
        self.temp = config.temp
        self.w_uniform = config.w_uniform  # 约束在全局交互数据在各个兴趣向量上分布均匀
        self.w_orth = config.w_orth  # 约束全局兴趣向量比较正交
        self.w_sharp = config.w_sharp  # 约束item属于一个全局兴趣向量
        self.w_clloss = config.w_clloss
        self.interest_type = config.interest_type
        self.add_cl = config.add_cl
        self.has_unique_loss = True
        self.cl_type = config.cl_type

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.n_interest = config.n_interest
        self.W = nn.Linear(self.embedding_size, self.embedding_size)
        self.selfatt_W = nn.Linear(self.n_interest, self.n_interest, bias=False)
        self.interest_embedding = nn.Embedding(self.n_interest, self.embedding_size)  # 设定为八个兴趣方向
        self.temperature = 0.1

        # GRU
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size, bias=True),
            nn.ReLU()
        )

        self.loss_fct = nn.CrossEntropyLoss()

        if 'Agg' in self.interest_type:
            self.interest_agg_layer = AttLayer(self.hidden_size)
        if 'Double' in self.interest_type:
            self.interest_agg_layer = AttLayer(self.hidden_size)
            self.interest_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 5, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid()
            )

        self.init_parameter()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)  # 参数的正则项系数
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    def init_parameter(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if name == 'interest_embedding':
                torch.nn.init.orthogonal_(weight.data)
                print(name)
            else:
                weight.data.uniform_(-stdv, stdv)

    def get_orth_loss(self, x):
        '''
        Args:
            x: batch_size * embed_size; Orthogonal embeddings
        Returns:
        '''
        num, embed_size = x.shape
        sim = x.reshape(-1, embed_size).matmul(x.reshape(-1, embed_size).transpose(0, 1))
        try:
            diff = sim - trans_to_cuda(torch.eye(sim.shape[1]))
        except RuntimeError:
            print('hello')
        regloss = diff.pow(2).sum() / (num * num)
        return regloss

    def forward(self, uids, item_seq, item_seq_len, istrain=False):
        with torch.no_grad():
            w = self.item_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.item_embedding.weight.copy_(w)
            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)
            w = self.user_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.user_embedding.weight.copy_(w)

        batch_size, n_seq = item_seq.shape
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        user_emb = self.user_embedding(uids)

        psnl_interest = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1,
                                                                           1)  # bs * n_interest * embed_size
        interest_cl = self.w_orth * self.get_orth_loss(self.interest_embedding.weight)

        for i in range(1):  # 迭代次数可以变成超参数
            scores = item_seq_emb.matmul(psnl_interest.transpose(1, 2)) / self.temp
            scores = scores.reshape(batch_size * n_seq, -1)
            mask = (item_seq > 0).reshape(-1)

            probs = torch.softmax(scores.reshape(batch_size, n_seq, -1), dim=-1) * (item_seq > 0).float().unsqueeze(-1)

            if self.w_uniform:
                interest_prb_vec = torch.sum(probs.reshape(batch_size * n_seq, -1), dim=0) / torch.sum(
                    mask)  # n_interest 1-dim vector
                # print(probs.shape, interest_prb_vec.shape)
                interest_cl += self.w_uniform * interest_prb_vec.std() / interest_prb_vec.mean()
                # todo: 求和均匀向量的交叉熵

            psnl_interest = probs.transpose(1, 2).matmul(item_seq_emb)
            psnl_interest = F.normalize(psnl_interest, dim=-1, p=2)

            sys_interest_vec = self.interest_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            interest_mask = torch.sum(probs, dim=1)  # batch_size * n_interest
            psnl_interest = torch.where(interest_mask.unsqueeze(-1) > 0, psnl_interest,
                                        sys_interest_vec)  # todo: 这里可以设置一个阈值
            batch_size, seq_len, n_interest = probs.shape

        # add global psnl embedding with GRU，用户对物品的偏好分数 = 某个单独的兴趣对物品的偏好分数 + 全局个性化偏好对物品的偏好分数
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.mlp(gru_output)
        full_psnl_emb = self.gather_indexes(gru_output, item_seq_len - 1)
        full_psnl_emb = F.normalize(full_psnl_emb, p=2, dim=-1)

        # 计算用户整体兴趣向量与各个兴趣点之间的相关性 interest importance scores
        # imp_probs = torch.softmax(
        #     full_psnl_emb.unsqueeze(1).matmul(psnl_interest.transpose(1, 2)).squeeze() / self.temp, dim=-1)
        # interest_mask = imp_probs  # 将 interest_mask 用于表示各个兴趣向量的重要程度

        if 'Agg' in self.interest_type:
            psnl_interest = self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(1)
        if 'Double' in self.interest_type:
            # calculate gate scores
            # agg_interest =  self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(1)
            # gate_input = torch.cat([agg_interest.repeat(1, self.n_interest, 1), psnl_interest, agg_interest-psnl_interest, agg_interest+psnl_interest, agg_interest*psnl_interest], dim=-1)
            # psnl_interest = agg_interest  + self.interest_gate(gate_input) * psnl_interest # 加上通过 attention 聚合的多兴趣
            psnl_interest = self.interest_agg_layer(full_psnl_emb, psnl_interest).unsqueeze(
                1) + psnl_interest  # 加上通过 attention 聚合的多兴趣

        '''开始计算最终的用户嵌入'''
        if istrain:
            if 'Plus' in self.interest_type and 'Uid' in self.interest_type:
                psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1) + user_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
            elif 'Plus' in self.interest_type:
                psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
            elif 'Uid' in self.interest_type:
                psnl_interest = psnl_interest + user_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
            return psnl_interest, interest_cl, interest_mask, full_psnl_emb
        else:
            if 'Plus' in self.interest_type and 'Uid' in self.interest_type:
                psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1) + user_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
            elif 'Plus' in self.interest_type:
                psnl_interest = psnl_interest + full_psnl_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest, p=2, dim=-1)
            elif 'Uid' in self.interest_type:
                psnl_interest = psnl_interest + user_emb.unsqueeze(1)
                psnl_interest = F.normalize(psnl_interest + user_emb.unsqueeze(1), p=2, dim=-1)
            return psnl_interest, interest_mask

    def get_userab_clloss(self, user_embed_a, user_embed_b):
        '''

        Args:
            user_embed_a: batch_size * embed_size
            user_embed_b: batch_size * embed_size

        Returns:

        '''
        batch_size, embed_size = user_embed_a.shape
        user_embed_a = F.normalize(user_embed_a, p=2, dim=-1).reshape(batch_size // 2, 2, embed_size)
        user_embed_b = F.normalize(user_embed_b, p=2, dim=-1).reshape(batch_size // 2, 2, embed_size)
        user_embed_aa = user_embed_a[:, 0]
        user_embed_ab = user_embed_a[:, 1]
        user_embed_ba = user_embed_b[:, 0]
        user_embed_bb = user_embed_b[:, 1]
        sim_matrix = torch.matmul(user_embed_aa, user_embed_ba.transpose(0, 1)) / self.temperature
        cl_loss = F.cross_entropy(sim_matrix, trans_to_cuda(torch.arange(batch_size // 2))) + F.cross_entropy(
            sim_matrix.transpose(0, 1), trans_to_cuda(torch.arange(batch_size // 2)))

        sim_matrixb = torch.matmul(user_embed_ab, user_embed_bb.transpose(0, 1)) / self.temperature
        cl_loss += F.cross_entropy(sim_matrixb, trans_to_cuda(torch.arange(batch_size // 2))) + F.cross_entropy(
            sim_matrixb.transpose(0, 1), trans_to_cuda(torch.arange(batch_size//2))
        )
        return cl_loss

    def get_user_clloss(self, user_embed):
        '''
        :param user_embed: (2*batch_size) * embed_size
        :return:
        '''
        batch_size, embed_size = user_embed.shape
        user_embed = F.normalize(user_embed, p=2, dim=-1)
        user_embed = user_embed.reshape(batch_size // 2, 2, embed_size)
        user_embed_a = user_embed[:, 0]
        user_embed_b = user_embed[:, 1]
        sim_matrix = torch.matmul(user_embed_a, user_embed_b.transpose(0, 1)) / self.temperature  # 之前忘了除这个温度系数
        cl_loss = F.cross_entropy(sim_matrix, trans_to_cuda(torch.arange(batch_size // 2))) + \
                  F.cross_entropy(sim_matrix.transpose(0, 1), trans_to_cuda(torch.arange(batch_size // 2)))
        return cl_loss

    def multi_inter_clloss(self, user_interests):
        '''
        下标 0 和 1 是同一个用户数据增强 后所学的不同兴趣，2 和 3 是同一个用户，以此类推，同一个用户同一兴趣之间是正样本，不同用户或不同兴趣之间是负样本
        Args:
            user_interests: batch_size * n_interest * embed_size
        Returns: loss
        '''
        batch_size, n_interest, embed_size = user_interests.shape
        user_interests = user_interests.reshape(batch_size // 2, 2, n_interest, embed_size)
        user_interests_a = user_interests[:, 0].reshape(-1, embed_size)
        user_interests_b = user_interests[:, 1].reshape(-1, embed_size)
        user_interests_a = F.normalize(user_interests_a, p=2, dim=-1)
        user_interests_b = F.normalize(user_interests_b, p=2, dim=-1)
        sim_matrix = user_interests_a.matmul(user_interests_b.transpose(0, 1)) / self.temperature
        loss = F.cross_entropy(sim_matrix, trans_to_cuda(torch.arange(sim_matrix.shape[0]))) + F.cross_entropy(
            sim_matrix.transpose(0, 1), trans_to_cuda(torch.arange(sim_matrix.shape[0])))
        return loss

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_all_item_label(self):
        ''' get hard label of each item'''
        with torch.no_grad():
            # w = self.item_embedding.weight.data.clone()
            # w = F.normalize(w, dim=-1, p=2)
            # self.item_embedding.weight.copy_(w)
            w = self.interest_embedding.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.interest_embedding.weight.copy_(w)

        item_seq_emb = self.item_embedding.weight  # n_item * embed_size
        interest_emb = self.interest_embedding.weight  # n_interest
        scores = torch.matmul(item_seq_emb, interest_emb.transpose(0, 1))  # n_item * n_interest
        labels = torch.argmax(scores, dim=-1)
        return labels

    def get_prob_kl(self, probs):
        '''
        Args:
            probs: batch_size * n_interest
        Returns:
        '''

        batch_size, n_interest = probs.shape
        probs = probs.reshape(batch_size // 2, 2, n_interest)

        # 使用两个正样本对分布的向量内积/余弦相似度来衡量相似度，然后使用对比损失
        prob_norm1 = F.normalize(probs[:, 0], dim=-1, p=2)
        prob_norm2 = F.normalize(probs[:, 1], dim=-2, p=2)
        scores = prob_norm1.matmul(prob_norm2.transpose(0, 1))  # batch_size * batch_size
        loss = F.cross_entropy(scores, trans_to_cuda(torch.arange(batch_size // 2)))

        # 如果用 kl 散度的话需要保证预测值进行过 softmax+log, 真实值进行过 softmax
        # loss = F.kl_div(probs[:, 0], probs[:, 1], reduction='mean') + F.kl_div(probs[:, 1], probs[:, 0], reduction='mean')
        return loss

    def calculate_loss(self, uids, item_seq, item_seq_len, pos_items, neg_items):
        psnl_user_embeds, interest_reg, interest_mask, full_user_embed = self.forward(uids, item_seq, item_seq_len,
                                                                                      istrain=True)
        batch_size, n_interest, embed_size = psnl_user_embeds.shape

        pos_items_emb = self.item_embedding(pos_items)
        neg_items_emb = self.item_embedding(neg_items)
        pos_scores = torch.sum(psnl_user_embeds * pos_items_emb.unsqueeze(1), dim=-1)
        neg_scores = psnl_user_embeds.reshape(-1, embed_size).matmul(neg_items_emb.transpose(0, 1)).reshape(
            batch_size, -1, batch_size)
        scores = torch.cat([pos_scores.unsqueeze(-1), neg_scores], dim=-1)

        # todo: 约束在推荐的时候，每个候选 item 尽可能只与用户的一个兴趣相关
        # scores: batch_size * n_interest * n_item
        # if self.has_unique_loss:
        # unique_scores = scores.transpose(1, 2).reshape(-1, scores.shape[1])
        # interest_reg += 0.01 * F.cross_entropy(unique_scores / self.temp, torch.argmax(unique_scores, dim=-1))
        # interest_reg += F.cross_entropy(unique_scores / self.temp, torch.argmax(scores, dim=-1))
        scores = torch.max(scores, dim=1)[0]
        loss = self.loss_fct(scores / self.temp, trans_to_cuda(torch.zeros(batch_size).long()))
        full_clloss = self.get_user_clloss(full_user_embed)
        # loss += 0.1 * full_clloss
        if self.add_cl:
            # if 'emb' in self.cl_type:
            #     multi_clloss = self.multi_inter_clloss(psnl_user_embeds)
            #     loss += self.w_clloss * multi_clloss
            # elif 'pro' in self.cl_type:
            #     multi_clloss = self.get_prob_kl(interest_mask)
            #     loss += self.w_clloss * multi_clloss
            # if 'emb' in self.cl_type:
            multi_clloss = self.multi_inter_clloss(psnl_user_embeds)
            loss += self.w_clloss * multi_clloss
            # 加上整体偏好的嵌入对比损失
            # full_clloss = self.get_user_clloss(full_user_embed)
            # loss += self.w_clloss * full_clloss
            # 累加用户的多个兴趣向量，然后约束相似
            # total_user_embed = torch.sum(psnl_user_embeds, dim=1)
            # total_clloss = self.get_user_clloss(total_user_embed)
            # loss += self.w_clloss * total_clloss
            # # 累加多个兴趣的用户嵌入与GRU所学习到的嵌入，约束相似
            # total_user_embed = torch.sum(psnl_user_embeds, dim=1)
            # loss += self.w_clloss * self.get_userab_clloss(total_user_embed, full_user_embed)

            # elif 'pro' in self.cl_type:
            # multi_clloss = self.get_prob_kl(interest_mask)
            # loss += self.w_clloss * multi_clloss
        return interest_reg + loss
