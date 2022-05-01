#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# NER模型（们）：BiLSTM-Linear, BiLSTM-Linear-CRF, BERT-BiLSTM-Linear-CRF, BERT(Prompt)
 
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM, AutoTokenizer
from crf_layer import CRFLayer

class INERModel(nn.Module):
    '''
    Interface of NER Models.
    '''
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
    
    def return_loss_by_logsoftmax_logits(self, logits, tags):
        if not tags is None:
            loss = F.nll_loss(F.log_softmax(logits, dim=1), tags, ignore_index=-100)
        else:
            loss = 0
        return {
            "logits" : logits.unsqueeze(0),
            "loss" : loss
        }
    
    def decode(self, **X):
        '''
        Decode by argmax-softmax
        X : { "input_ids" : [batch_size, seq_len] }
        '''
        logits = self(**X)["logits"]
        return torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()


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
        batch_size should be 1
        '''
        embed : torch.Tensor = self.embed(X["input_ids"])                       #[1, seq_len, embedding_size]
        lstm_out, _ = self.lstm(embed)                                          #[1, seq_len, lstm_hidden_size * 2]
        logits = self.linear(self.dropout(lstm_out.squeeze(0)[1:-1, :]))        #[seq_len, tag_size]
        return self.return_loss_by_logsoftmax_logits(logits, X["tags"].view(-1)[1:-1] if "tags" in X else None)
    
        
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
        X = { "input_ids" : [batch_size, seq_len], "tags" : [batch_size, seq_len] }
        Restrict: batch_size = 1
        '''
        if "tags" in X:
            tags = X.pop("tags")
            logits = self.encoder(**X)["logits"]
            loss = self.crf.nll_loss(logits.squeeze(0), tags.squeeze(0)[1:-1])
            X["tags"] = tags
            return {
                "logits" : logits,
                "loss" : loss
            }
        else:
            return self.encoder(**X)

    def decode(self, **X):
        '''
        X : { "input_ids" : [batch_size, seq_len] }
        '''
        logits = self.encoder(**X)["logits"].squeeze(0)
        return [self.crf.decode(logits)]


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
        '''
        bert_model : bert模型(transformers.BertModel或bert模型的url)
        tag_size : 标签集大小
        dropout_ratio : dropout率
        '''
        super().__init__()
        if isinstance(bert_model, BertModel):
            self.bert = bert_model
        elif isinstance(bert_model, str):
            self.bert = BertModel.from_pretrained(bert_model)
        else:
            raise RuntimeError("Can't load bert_model")
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(self.bert.config.hidden_size, tag_size)
        self.tag_size = tag_size

    def forward(self, **X):
        '''
        X : { "input_ids" : [batch_size, seq_len], "tags" : [batch_size, seq_len] }
        '''
        bert_out : torch.Tensor = self.bert(input_ids=X["input_ids"])[0][:,1:-1,:].squeeze(0)   #[seq_len, hidden_size]
        bert_out = self.dropout(bert_out)
        logits = self.linear(bert_out)
        return self.return_loss_by_logsoftmax_logits(logits, X["tags"].view(-1)[1:-1] if "tags" in X else None)


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
        '''
        X : { "input_ids" : [batch_size, seq_len], "tags" : [batch_size, seq_len] }
        batch_size = 1
        '''
        bert_out = self.bert(input_ids=X["input_ids"])[0]                 #[batch_size, seq_len, bert_hidden_size]
        lstm_out, _ = self.lstm(bert_out)                                 #[batch_size, seq_len, lstm_hidden_size]
        logits = self.linear(self.dropout(lstm_out[:, 1:-1, :]))  #[seq_len, tag_size]
        return self.return_loss_by_logsoftmax_logits(logits.squeeze(0), X["tags"].view(-1)[1:-1] if "tags" in X else None)


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
    

class NER_BERT_Prompt(INERModel):
    def __init__(self, bert_model, tag_names):
        super().__init__()
        if isinstance(bert_model, BertForMaskedLM):
            self.bert = bert_model
            self.bert_name = self.bert.config._name_or_path
        elif isinstance(bert_model, str):
            self.bert = BertForMaskedLM.from_pretrained(bert_model)
            self.bert_name = bert_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.tag_names = list(tag_names)
        self.tag2idx = {}
        for i, tag in enumerate(tag_names):
            self.tag2idx[tag] = i
        
    def forward(self, **X):
        return self.bert(input_ids=X["input_ids"], labels=X["tags"], return_dict=True)

    def decode_to_tags(self, **X):
        logits = self.forward(**X)['logits'].squeeze(0)[1:-1]
        result = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
        
        dec = []
        for tid in result:
            word = self.tokenizer.decode(tid)
            if word in self.tag_names:
                dec.append(word)
            else:
                dec.append('O')
        return [dec]
   

    def decode(self, **X):
        dec = self.decode_to_tags(**X)
        return [
            [self.tag2idx[t] for t in tags] for tags in dec
        ]

        


