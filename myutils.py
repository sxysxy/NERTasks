#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: GPL-v3
# 一些通用的东西

from datasets import load_dataset, load_metric
import os

class Configs:
    '''
    配置
    '''
    def __init__(self, namespec):
        self.dataset_name : str = namespec.dataset
        self.ner_epoches : int = namespec.ner_epoches
        self.warmup_ratio : float = namespec.warnmup_ratio
        self.ner_lr : float = namespec.ner_lr
        self.use_bert : bool = namespec.use_bert
        self.bert_name_or_path : str = namespec.bert_name_or_path
        self.label_smooth_ratio : float = namespec.label_smooth_ratio
        self.dropout_ratio : float = namespec.dropout_ratio
        self.lstm_layers : float = namespec.lstm_layers
        self.lstm_hidden_size : float = namespec.lstm_hidden_size
        self.embedding_size : float = namespec.embedding_size
        
    def __getitem__(self, idx):
        return object.__getattribute__(self, idx)


class NERTokenizerFromDataset:
    '''
    transformers.BertTokenizer style Tokenzer
    但并不from_ptrtrained，只是通过某一个数据集来构成词表
    '''
    def __init__(self) -> None:
        self.word2idx = {}
        self.idx2word = []

    def rebuild_vocab(self, texts : list[list[str]]):
        self.word2idx = {
            self.pad_token : self.pad_token_id,
            self.cls_token : self.cls_token_id,
            self.sep_token : self.sep_token_id,
            self.unk_token : self.unk_token_id
        }
        self.idx2word = [self.pad_token, self.cls_token, self.sep_token, self.unk_token]
        
        n = len(self.word2idx)
        for text in texts:
            for word in text:
                if not word in self.word2idx:
                    self.word2idx[word] = n 
                    self.idx2word.append(word)
                    n += 1

    def batch_encode_plus(self, texts : list[list[str]], add_special_tokens=True, padding = True, truncation = True, max_length = 512,
                                                 return_length = True, is_split_to_words = True):
        '''
        类似于transformers.BertTokenizer.batch_encode_plus
        由于针对NER任务，所以is_split_to_words参数实际上是忽略的，是默认有效的，只是为了和transformers的保持接口的兼容
        一定会返回length和attention_mask
        一定会返回word_ids
        '''
        N = len(texts)
        input_ids = []
        attention_mask = []
        length = []
        word_ids = []
        for t in texts:
            ids = []
            if add_special_tokens:
                ids.append(self.cls_token_id)
            if truncation:
                valid_t = t[:max_length-2] if add_special_tokens else t[:max_length]
            else:
                valid_t = t
            for w in valid_t:
                ids.append(self.word2idx.get(w, self.unk_token_id))
            if add_special_tokens:
                ids.append(self.sep_token_id)
            real_len = len(ids)

            assert real_len <= max_length

            length.append(real_len)
            if padding:
                if real_len < max_length:
                    ids += [self.pad_token_id] * (max_length - real_len)
                    attention_mask.append( ([1] * real_len) + [0] * (max_length - real_len) )
                else:
                    attention_mask.append( [1] * real_len )
            else:
                attention_mask.append( [1] * real_len )

            input_ids.append(ids)
            word_ids.append(list(range(real_len)))
        return {
            "input_ids" : input_ids,
            "word_ids" : word_ids,
            "length" : length,
            "attention_mask" : attention_mask
        }
            
    def batch_decode(self, batch_ids : list[list[int]], ignore_special_tokens = False):    
        texts = []
        for ids in batch_ids:
            text = []
            for id in ids:
                if not (ignore_special_tokens and id <= self.sep_token_id):  #[UNK]不认为是特殊符号
                    text.append(self.idx2word[id])
            texts.append(text)
        return texts

    def __call__(self, **kwargs):
        return self.batch_encode_plus(**kwargs)
    
    @property
    def vocab_size(self):
        return len(self.word2idx)

    @property
    def pad_token_id(self):
        return 0
    @property
    def pad_token(self):
        return "[PAD]"
    @property
    def cls_token_id(self):
        return 1
    @property
    def cls_token(self):
        return "[CLS]"
    @property
    def sep_token_id(self):
        return 2
    @property
    def sep_token(self):
        return "[SEP]"
    @property
    def unk_token_id(self):
        return 3
    @property
    def unk_token(self):
        return "[UNK]"

def get_current_full_dirname():
    return os.path.split(__file__)[0]

def load_datasets(name):
    return load_dataset(f"{get_current_full_dirname()}/load_datasets.py", name)

os.environ["raw_datasets_path"] = f"{get_current_full_dirname()}/assets/raw_datasets"