#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# NER模型（们）：BiLSTM-Linear, BiLSTM-Linear-CRF, BERT-BiLSTM-Linear-CRF, BERT(Prompt)
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from crf_layer import CRFLayer

class INERModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, X):
        '''
        这只是个示例
        X至少应当包含 "input_ids"
        '''
        return {
            "logits" : torch.tensor(0).to(self.device),
            "loss" : torch.tensor(0).to(self.device)
        }

    def nll_loss(self, logits, tag_space_size, X):
        '''
        用于计算一批样本X的 nll_loss。这个函数对于批量的样本很有用，对于使用了CRF的模型可能不需要用到。
        logits : logits
        tag_space_size : 标签种类数
        X : 一批样本
        '''
        labels = X["tags"]
        if "attention_mask" in X:
            attention_mask = X["attention_mask"]
            active_loss = attention_mask.view(-1) == 1
            active_logits = torch.log_softmax(logits.view(-1, tag_space_size), dim=1)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(-100).type_as(labels)  #-100是ignore_idx
            )
            loss = F.nll_loss(active_logits, active_labels)
        else:
            loss = F.nll_loss(torch.log_softmax(logits.view(-1, tag_space_size), dim=1), labels.view(-1))
        return loss

    def make_result(self, logits, tag_size, X, calc_loss = True):
        '''
        返回logits
        如果X内包含tags，则也返回loss
        返回值形式是字典 {"logits" : xxx, "loss" : xxx}
        logits : logits
        tag_size : 标签种类数
        X : batch
        calc_loss : 是否计算loss
        '''
        out = {
            "logits" : logits
        }
        if calc_loss:
            out["loss"] = self.nll_loss(logits, tag_size, X)
        return out

    def decode(self, X):
        '''
        softmax解码
        不适用于CRF
        '''
        logits = self(X)['logits']
        return F.softmax(logits, dim=-1)

    
class NER_BiLSTM_Linear(INERModel):
    '''
    BiLSTM-CRF的模型
    '''
    def __init__(self, vocab_size, tag_size, embedding_size, lstm_layers, lstm_hidden_size, dropout_ratio):
        '''
        vocab_size: 词表大小
        tag_size: 标签集大小
        embedding_size: 词嵌入维度
        lstm_layers: LSTM层数
        lstm_hidden_size: LSTM隐藏层大小
        droupout_ratio: dropout率
        '''
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_hidden_size, 
                            batch_first=True,
                            bidirectional=True,
                            num_layers=lstm_layers)
        self.droupout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(lstm_hidden_size * 2, tag_size)
        self.tag_size = tag_size
    
    def forward(self, X : dict):
        '''
        X = { "input_ids" : [batch_size, seq_len], "attention_mask" : [batch_size, seq_len], "tags" : [batch_size, seq_len] }
        '''
        input_ids = X["input_ids"]                          #[batch_size, seq_len]
        embed = self.embed(input_ids)                       #[batch_size, seq_len, embedding_size]
        lstm_out, _ = self.lstm(embed)                      #[batch_size, seq_len, 2 * hidden_size]
        logits = self.linear(self.droupout(lstm_out))       #[batch_size, seq_len, tag_size)
        return self.make_result(logits, self.tag_size, X, "tag" in X)   
        

class NER_BiLSTM_Linear_CRF(INERModel):
    '''
    BiLSTM-Linear-CRF
    使用CRF模型对标签序列建模，并使用CRF评估标注序列的负对数似然或解码
    '''
    def __init__(self, vocab_size, tag_size, embedding_size, lstm_layers, lstm_hidden_size, dropout_ratio):
        '''
        vocab_size: 词表大小
        tag_size: 标签集大小
        embedding_size: 词嵌入维度
        lstm_layers: LSTM层数
        lstm_hidden_size: LSTM隐藏层大小
        droupout_ratio: dropout率
        '''
        super().__init__()
        self.encoder = NER_BiLSTM_Linear(vocab_size, tag_size, embedding_size, lstm_layers, lstm_hidden_size, dropout_ratio)
        self.crf = CRFLayer(tag_size)
        self.tag_size = tag_size

    def forward(self, X : dict):
        '''
        X = { "input_ids" : [batch_size, seq_len], "attention_mask" : [batch_size, seq_len], "tags" : [batch_size, seq_len], "length" : [batch_size] }
        '''
        if "tags" in X:
            tags = X.pop("tags")
            logits = self.encoder(X)["logits"]
            losses = []
            for (feat, label, len) in zip(logits, tags, X["length"]):           #这个地方没法并行计算就慢了
                losses.append(self.crf.nll_loss(feat[:len, :], label[:len]))
            loss = losses[0]
            for i in range(1, len(losses)):
                loss += losses[i]
            X["tags"] = tags
            return {
                "logits" : logits,   #这个logits并不是CRF的输出，是BiLSTM-Linear的
                "loss" : loss
            } 
        else:
            return self.encoder(X)

    def decode(self, X : dict):
        '''
        X 至少包含input_ids和length，length给出真实的长度
        '''
        results = []
        logits = self.encoder(X)["logits"]
        for (feat, len) in zip(logits, X["length"]):
            results.append(self.crf.decode(feat[:len]))
        return results
        

    

