#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn)
# License: LGPL-v3
# 线性条件随机场模型 CRF
# 学习自 https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?highlight=crf

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def batch_log_sum_exp(vec):
    '''
    vec: [N, D]
    N个D维向量, 对每个向量做log_sum_exp
    返回: [N], 每一个向量的log_sum_exp的值。
    速度更快, 产生计算图节点更少。
    '''
    max_score = torch.max(vec, dim=1).values
    max_score_broadcast = max_score.view(-1, 1)
    '''
    数值稳定性更好
    \mathrm{log}{\sum_i e^i} = k + \mathrm{log}{\sum_i e^{i-k}}
    '''
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

class CRFLayer(nn.Module):
    def __init__(self, tag_size, required_grad = True) -> None:
        super().__init__()
        #设trans[i,j]表示从标签j到标签i的转移分数
        #相比于“从i到j”的意义，可以省去N个转置操作或者列切片操作（行优先存储的话列切片慢一点点？），速度更快
        self.trans = nn.Parameter(torch.zeros(tag_size + 2, tag_size + 2, dtype=float), requires_grad=required_grad)
        self.start_tag_id = tag_size 
        self.end_tag_id = tag_size + 1
        self.trans.data[self.start_tag_id, :] = -10000.0    #任意标签转移到START_TAG，分数-10000
        self.trans.data[:, self.end_tag_id] = -10000.0      #从END_TAG转移到任意标签，分数-10000
        self.tag_size = tag_size
        self.real_tag_size = self.tag_size + 2

    #计算分子exp(s(X, y))的log，其实就是s(X, y) = trans + emit
    def log_exp_score(self, X : torch.Tensor, y : torch.Tensor):
        '''
        X : [L, tag_size], lstm_linear的输出
        y : [L]
        '''
        score = torch.zeros(1).to(X.device)
        for i, emit in enumerate(X):
            score += self.trans[y[i], y[i-1] if i > 0 else self.start_tag_id] + emit[y[i]]
        score += self.trans[self.end_tag_id, y[-1]]
        return score

    #计算分母\sum_{\hat{y}} exp(s(X, \hat{y}))的log
    def log_exp_sum_score(self, X : torch.Tensor):
        '''
        X : [L, tag_size], lstm_linear的输出
        '''
        sum_score = torch.full((1, self.real_tag_size), -10000.0).to(X.device)
        sum_score[0][self.start_tag_id] = 0
        #原版是 https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?highlight=crf
        #这里我写的是我优化的版本，更短更快
        for emit in X:
            score = sum_score + self.trans + emit.unsqueeze(0).repeat(self.real_tag_size, 1).transpose(0, 1)
            sum_score = batch_log_sum_exp(score).view(1, -1)
        sum_score += self.trans[self.end_tag_id]
        return batch_log_sum_exp(sum_score)[0]

    #目标函数的负对数似然
    def nll_loss(self, X : torch.Tensor, y : torch.Tensor):
        return self.log_exp_sum_score(X) - self.log_exp_score(X, y)

    #解码，这个得到的结果不能用于交叉熵损失函数，因为我no_grad()了~ 训练用上面的nll_loss
    def decode(self, X : torch.Tensor) -> list[int]:
        with torch.no_grad():
            stack = []
            sum_score = torch.full((1, self.real_tag_size), -10000.0).to(X.device)
            sum_score[0][self.start_tag_id] = 0
            #原版是 https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?highlight=crf
            #这里我写的是我优化的版本，更短更快
            for emit in X:
                score = sum_score + self.trans
                best_tags = torch.argmax(score, dim=1)
                sum_score = score.gather(1, best_tags.view(-1, 1)).view(1, -1) + emit
                stack.append(best_tags)
            sum_score += self.trans[self.end_tag_id]
            best_tag_id = torch.argmax(sum_score, dim=1).item()
            path = [best_tag_id]
            for node in reversed(stack):
                best_tag_id = node[best_tag_id].item()
                path.append(best_tag_id)
            start = path.pop()
            assert start == self.start_tag_id
            path.reverse()
            return path
            #return np.array(path, dtype=int) 

    def init_transitions(self, trans : torch.Tensor, require_grad=True):
        self.trans = nn.Parameter(trans, requires_grad=require_grad)
    