#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# NER模型（们）：BiLSTM-Linear, BiLSTM-Linear-CRF, BERT-BiLSTM-Linear-CRF, BERT(Prompt)
 
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from crf_layer import CRFLayer

class INERModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, **X):
        '''
        这只是个示例
        return {
            "logits" : torch.tensor(0).to(self.device),
            "loss" : torch.tensor(0).to(self.device)
        }
        '''
        raise RuntimeError("You should not come here")
    
    def decode(self, **X):
        '''
        解码一批样本
        '''
        raise RuntimeError("You should not come here")
    

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
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(lstm_hidden_size * 2, tag_size)
        self.tag_size = tag_size
    
    def forward(self, **X):
        '''
        X = { "input_ids" : [batch_size, seq_len], "attention_mask" : [batch_size, seq_len], "tags" : [batch_size, seq_len] }
        '''
        calc_loss = "tags" in X
        embed = self.embed(X["input_ids"])                   #[batch_size, seq_len, embedding_size]
        losses = []
        logits = []
        if calc_loss:
            for seq, tag, seq_len in zip(embed, X["tags"], X["length"]):
                lstm_out, _ = self.lstm(seq[:seq_len, :].unsqueeze(0))
                logit = self.linear(self.dropout(lstm_out.squeeze(0)))
                loss = F.nll_loss(F.log_softmax(logit, dim=1), tag[:seq_len], ignore_index=-100)
                losses.append(loss.view(-1))
                logits.append(logit)
        else:
            for seq, seq_len in zip(embed, X["length"]):
                lstm_out, _ = self.lstm(seq[:seq_len, :].unsqueeze(0))
                logit = self.linear(self.dropout(lstm_out.squeeze(0)))
                logits.append(logit)

        out = {"logits" : logits}
        if calc_loss:
            out["loss"] = torch.cat(losses).mean()
        return out
        #lstm_out, _ = self.lstm(embed)                      #[batch_size, seq_len, 2 * hidden_size]
        #logits = self.linear(self.droupout(lstm_out))       #[batch_size, seq_len, tag_size]
        #return self.make_result(logits, self.tag_size, X, "tags" in X)   
    
    def decode(self, **X):
        logits = self(**X)["logits"]
        results = []
        for logit in logits:
            results.append(torch.argmax(F.softmax(logit, dim=1), dim=1).tolist())
        return results
        
class NER_With_CRF(INERModel):
    '''
    把模型的crf之前部分看作是一个encoder，CRF看作是decoder
    '''
    def __init__(self, encoder, tag_size) -> None:
        super().__init__()
        self.encoder = encoder
        self.crf = CRFLayer(tag_size)
        self.tag_size = tag_size
    
    def forward(self, **X):
        '''
        X = { "input_ids" : [batch_size, seq_len], "attention_mask" : [batch_size, seq_len], "tags" : [batch_size, seq_len], "length" : [batch_size] }
        '''
        if "tags" in X:
            tags = X.pop("tags")
            logits = self.encoder(**X)["logits"]
            losses = []
            for (feat, label, seq_len) in zip(logits, tags, X["length"]):           #这个地方没法并行计算就慢了
                losses.append(self.crf.nll_loss(feat[1:seq_len-1, :], label[1:seq_len-1]))
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

    def decode(self, **X):
        '''
        X 至少包含input_ids和length，length给出真实的长度
        '''
        results = []
        logits = self.encoder(**X)["logits"]
        for (feat, seq_len) in zip(logits, X["length"]):
            results.append(self.crf.decode(feat[1:seq_len-1]))
        return results

class NER_BiLSTM_Linear_CRF(NER_With_CRF):
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
        encoder = NER_BiLSTM_Linear(vocab_size, tag_size+2, embedding_size, lstm_layers, lstm_hidden_size, dropout_ratio)
        super().__init__(encoder, tag_size)

class NER_BERT_Linear(INERModel):
    def __init__(self, bert_model, tag_size, dropout_ratio) -> None:
        super().__init__()
        if isinstance(bert_model, nn.Module):
            self.bert = bert_model
        elif isinstance(bert_model, str):
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            raise RuntimeError("Can't load bert_model")
        self.dropout = nn.Dropout(dropout_ratio)
        self.classifier = nn.Linear(self.bert.config.hidden_size, tag_size)
        self.tag_size = tag_size

     
    def forward(self, **X):
        '''
        X : {"input_ids" : [batch_size, seq_len],
             "attention_mask" : [batch_size, seq_len]
             "tags" : [batch_size, seq_len]
             }
        '''
        max_len = torch.max(X["length"])
        attention_mask : torch.Tensor = X["attention_mask"][:, :max_len]
        bert_out : torch.Tensor = self.bert(input_ids=X["input_ids"][:, :max_len], attention_mask=attention_mask)[0]  #[batch_size, seq_len, hidden_size]
        bert_out = self.dropout(bert_out)
        logits = self.classifier(bert_out)     #[batch_size, seq_len, tag_space_size]
        res = {
            "logits" : logits,  #[batch_size, seq_len, tag_space_size]
            "hidden_states" : bert_out,     
        }

        if "tags" in X:
            tags : torch.Tensor = X["tags"][:, :max_len]
            if not (attention_mask is None):
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = torch.log_softmax(logits.view(-1, self.tag_size), dim=1)
                active_labels = torch.where(
                    active_loss, tags.contiguous().view(-1), torch.tensor(-100).type_as(tags)  #-100是ignore_idx
                )
                loss = F.nll_loss(active_logits, active_labels)
            else:
                loss = F.nll_loss(torch.log_softmax(logits.view(-1, self.tag_size), dim=1), tags.contiguous().view(-1))
            res["loss"] = loss
        return res

    def decode(self, **X):
        tag_in_X = False
        if "tags" in X:
            tags = X.pop("tags")  #阻止forward内计算loss
            tag_in_X = True
        logits = self(**X)["logits"]
        result = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        if tag_in_X:
            X["tags"] = tags
        return result

class NER_BERT_BiLSTM_Linear(INERModel):
    def __init__(self, bert_model, tag_size, lstm_layers, lstm_hidden_size, dropout_ratio):
        '''
        bert_model : bert模型或者bert_name/path
        tag_size : 标签集大小
        lstm_layers : lstm层数
        lstm_hidden_size : lstm隐藏层大小
        dropout_ratio : dropout率
        '''
        super().__init__()
        if isinstance(bert_model, nn.Module):
            self.bert = bert_model
        elif isinstance(bert_model, str):
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            raise RuntimeError("Can't load bert_model")
        self.dropout = nn.Dropout(dropout_ratio)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=lstm_hidden_size, 
                            batch_first=True,
                            bidirectional=True,
                            num_layers=lstm_layers)
        self.linear = nn.Linear(2 * lstm_hidden_size, tag_size)
        self.tag_size = tag_size
    
    def forward(self, **X):
        calc_loss = "tags" in X
        bert_out = self.bert(X["input_ids"])[0]                   #[batch_size, seq_len, bert_hidden_size]
        losses = []
        logits = []
        if calc_loss:
            for seq, tag, seq_len in zip(bert_out, X["tags"], X["length"]):
                lstm_out, _ = self.lstm(seq[:seq_len, :].unsqueeze(0))
                logit = self.linear(self.dropout(lstm_out.squeeze(0)))
                loss = F.nll_loss(F.log_softmax(logit, dim=1), tag[:seq_len], ignore_index=-100)
                losses.append(loss.view(-1))
                logits.append(logit)
        else:
            for seq, seq_len in zip(bert_out, X["length"]):
                lstm_out, _ = self.lstm(seq[:seq_len, :].unsqueeze(0))
                logit = self.linear(self.dropout(lstm_out.squeeze(0)))
                logits.append(logit)

        out = {"logits" : logits}
        if calc_loss:
            out["loss"] = torch.cat(losses).mean()
        return out

    def decode(self, **X):
        logits = self(**X)["logits"]
        results = []
        for logit in logits:
            results.append(torch.argmax(F.softmax(logit, dim=1), dim=1).tolist())
        return results


class NER_BERT_BiLSTM_Linear_CRF(NER_With_CRF):
    def __init__(self, bert_model, tag_size, lstm_layers, lstm_hidden_size, dropout_ratio):
        '''
        bert_model : bert模型或者bert_name/path
        tag_size : 标签集大小
        lstm_layers : lstm层数
        lstm_hidden_size : lstm隐藏层大小
        dropout_ratio : dropout率
        '''
        encoder = NER_BERT_BiLSTM_Linear(bert_model, tag_size+2, lstm_layers, lstm_hidden_size, dropout_ratio)
        super().__init__(encoder, tag_size)