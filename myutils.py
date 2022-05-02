#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# 一些通用的东西


from typing import Tuple
from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
import os
import sys
import argparse
import pdb
import copy
from transformers import AutoTokenizer, AdamW
from typer import prompt
from ner_models import (
    NER_BERT_BiLSTM_Linear, NER_BERT_BiLSTM_Linear_CRF, 
    NER_BERT_Linear, NER_BERT_Prompt, NER_BiLSTM_Linear, 
    NER_BiLSTM_Linear_CRF, NER_BERT_Linear_CRF
)
import pickle
import ujson as json
import torch

class Configs:
    OVERALL_MICRO = 0
    OVERALL_MACRO = 1

    '''
    配置
    '''
    def __init__(self, namespec):
        self.dataset_name : str = namespec.dataset
        self.ner_epoches : int = namespec.ner_epoches
        self.warmup_ratio : float = namespec.warmup_ratio if namespec.warmup_ratio > 0 else None
        self.grad_acc : int = namespec.grad_acc
        self.batch_size : int = namespec.batch_size
        self.clip_grad_norm : float = namespec.clip_grad_norm if namespec.clip_grad_norm > 0 else None
        self.ner_lr : float = namespec.ner_lr
        self.ner_weight_decay : float = namespec.ner_weight_decay 
        self.model_name : str = namespec.model_name
        self.bert_name_or_path : str = namespec.bert_name_or_path
        self.label_smooth_factor : float = namespec.label_smooth_factor if namespec.label_smooth_factor > 0 else None
        self.dropout_ratio : float = namespec.dropout_ratio
        self.lstm_layers : float = namespec.lstm_layers
        self.lstm_hidden_size : float = namespec.lstm_hidden_size
        self.embedding_size : float = namespec.embedding_size
        self.random_seed : float = namespec.random_seed
        self.few_shot : float = namespec.few_shot
        self.f1 = {
            "overall_micro" : Configs.OVERALL_MICRO,
            "overall_macro" : Configs.OVERALL_MACRO
        }[namespec.f1]
        self.save_model : bool = namespec.save_model
    
    @classmethod
    def parse_from(cls, argv):
        ps = argparse.ArgumentParser()
        ps.add_argument("--dataset", choices=['conll2003', 'ontonotes5', 'cmeee'], type=str, required=True, 
                            help="Choose dataset.")
        ps.add_argument("--ner_epoches", type=int, default=10, 
                            help="The number of training epochs on NER Task, default = 10.")
        ps.add_argument("--warmup_ratio", type=float, default=0.2,
                            help="Warmup epoches / total epoches, default = 0.2.")
        ps.add_argument("--grad_acc", type=int, default=8, 
                            help="Gradient accumulation, default=8.")
        ps.add_argument("--batch_size", type=int, default=1, 
                            help="Batch size, default to 1.")
        ps.add_argument("--clip_grad_norm", type=float, default=1.0,
                            help="torch.nn.utils.clip_grad_norm_")
        ps.add_argument("--ner_lr", type=float, default=3e-4,
                            help="Learning rate, default = 3e-4.")
        ps.add_argument("--ner_weight_decay", type=float, default=5e-3,
                            help="L2 penalty, default = 5e-3.")
        ps.add_argument("--model_name", choices=['BiLSTM-Linear', 'BiLSTM-Linear-CRF', 
                                                 'BERT-BiLSTM-Linear-CRF', 'BERT-Linear', 'BERT-Linear-CRF', 
                                                 "BERT-BiLSTM-Linear",  
                                                 "BERT-Prompt"],
                            type=str, required=True,
                            help="For no bert, specific which model to use.")
        ps.add_argument("--bert_name_or_path", type=str, 
                            help="Bert name(eg. bert-base-uncased) or path(eg. /local/path/to/bert-base-chinese).")
        ps.add_argument("--label_smooth_factor", type=float, default=0.1, 
                            help="Label smooth factor. Default to 0.1.")
        ps.add_argument("--dropout_ratio", type=float, default=0.2,
                            help="Dropout ratio on lstm_out/bert_out) when training, default = 0.2.")
        ps.add_argument("--lstm_layers", type=int, default=2, 
                            help="The number of layers of bidrectional LSTM, default = 1.")
        ps.add_argument("--lstm_hidden_size", type=int, default=256,
                            help="nn.LSTM(hidden_size), default=256")
        ps.add_argument("--embedding_size", type=int, default=256,
                            help="nn.Embedding(), default = 256.")
        ps.add_argument("--random_seed", type=int, default=233,
                            help="Random seed for transformers.sed_seed, default = 233")   
        ps.add_argument("--few_shot", type=float, default=None,
                            help="Few shot: Use len(trainset) * few_shot samples to train. Default to None(Full data).")
        ps.add_argument("--f1", type=str, choices=["overall_micro", "overall_macro"], default="overall_micro",
                            help="Micro F1 or Macro F1?")
        ps.add_argument("--save_model", action="store_true", default=False,
                            help="Whether to save the best model during training")
        return ps.parse_args(argv)
    
    cached_config = None
    @classmethod
    def parse_from_argv(cls):
        if cls.cached_config:
            return Configs.cached_config

        cls.cached_config = Configs(Configs.parse_from(sys.argv[1:]))
        return cls.cached_config
        
    def __getitem__(self, idx):
        return object.__getattribute__(self, idx)

    @property
    def using_bert(self):
        return self.model_name in ["BERT-BiLSTM-Linear-CRF", "BERT-Prompt", "BERT-Linear", "BERT-Linear-CRF", "BERT-BiLSTM-Linear"]

    @property
    def using_prompt(self):
        return self.model_name in ["BERT-Prompt"]

    @property
    def __dict__(self):
        d = {}
        for k in ["dataset_name", "ner_epoches", "warmup_ratio", 
         "grad_acc", "batch_size", "clip_grad_norm", "ner_lr", "ner_weight_decay",
         "model_name", "bert_name_or_path", "label_smooth_factor", "dropout_ratio",
         "lstm_layers", "lstm_hidden_size", "embedding_size", "random_seed", 
         "few_shot", "f1"]:
            d[k] = self[k]
        return d


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
    
    class BatchEncodedData:
        def __init__(self, input_ids, attention_mask, length, word_ids) -> None:
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.length = length
            self.word_ids_ = word_ids
        def __getitem__(self, idx):
            return object.__getattribute__(self, idx)

        def word_ids(self, sample_idx_in_batch):
            '''
            取得本批数据中制定下标的数据的word_ids
            '''
            return self.word_ids_[sample_idx_in_batch]

    def batch_encode_plus(self, texts : list[list[str]], add_special_tokens=True, padding = True, truncation = True, max_length = 512,
                                                 return_length = True, is_split_into_words = True):
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
            wids = []
            if add_special_tokens:
                ids.append(self.cls_token_id)
                wids.append(None)
            if truncation:
                valid_t = t[:max_length-2] if add_special_tokens else t[:max_length]
            else:
                valid_t = t
            for j, w in enumerate(valid_t):
                ids.append(self.word2idx.get(w, self.unk_token_id))
                wids.append(j)

            if add_special_tokens:
                ids.append(self.sep_token_id)
                wids.append(None)

            real_len = len(ids)

            assert real_len <= max_length

            length.append(real_len)
            if padding:
                if real_len < max_length:
                    ids += [self.pad_token_id] * (max_length - real_len)
                    wids += [None] * (max_length - real_len)
                    attention_mask.append( ([1] * real_len) + [0] * (max_length - real_len) )
                else:
                    attention_mask.append( [1] * real_len )
            else:
                attention_mask.append( [1] * real_len )

            input_ids.append(ids)
            word_ids.append(wids)

        return self.BatchEncodedData(input_ids, attention_mask, length, word_ids)
            
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
    def vocab(self):
        return self.word2idx
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
    @property
    def do_lower_case(self):
        return False
 
    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "tokenizer.bin"), "rb") as f:
            data = pickle.load(f)
            tk = NERTokenizerFromDataset()
            tk.word2idx = data["word2idx"]
            tk.idx2word = data["idx2word"]
            return tk

    def save_pretrained(self, path):
        os.makedirs(path, mode=0o755, exist_ok=True)
        with open(os.path.join(path, "tokenizer.bin"), "wb") as f:
            pickle.dump({
                "word2idx" : self.word2idx,
                "idx2word" : self.idx2word
            }, f)    

def get_base_dirname():
    return os.path.split(__file__)[0]

def get_datasets(name):
    return load_dataset(f"{get_base_dirname()}/load_datasets.py", name)

def get_ner_evaluation():
    return load_metric(f"{get_base_dirname()}/ner_evaluation.py")

os.environ["assets_path"] = f"{get_base_dirname()}/assets"

def dataset_map_raw2ner_kernel(tokenizer, examples):
    tokenized = tokenizer.batch_encode_plus(examples["tokens"], add_special_tokens=True, is_split_into_words=True, 
                                            padding=False, max_length=512, truncation=True, return_length=True)
    batch_tags = []
    length = []
    for sample_i, tags in enumerate(examples["tags"]):
        wids = tokenized.word_ids(sample_i)
        real_tags = []
        for wid in wids:
            if wid is None:
                real_tags.append(-100)  #Ignore index for pytorch
            else:
                real_tags.append(tags[wid])
        batch_tags.append(real_tags)
        length.append(sum(tokenized["attention_mask"][sample_i]))
    
    return {
        "input_ids" : [torch.tensor(a, dtype=torch.long) for a in tokenized["input_ids"]],
        "tags" : [torch.tensor(a, dtype=torch.long) for a in batch_tags]
    }

def dataset_map_raw2ner(dataset : DatasetDict, tokenizer) -> Tuple[DatasetDict, list[str]]:
    columns = ["input_ids", "tags"]
    ds = dataset.map(lambda x : dataset_map_raw2ner_kernel(tokenizer, x), batched=True)
    ds.set_format(type='torch', columns=columns)
    return ds, columns

# def dataset_map_raw2ner_padded(tokenizer, examples):
#     tokenized = tokenizer.batch_encode_plus(examples["tokens"], add_special_tokens=True, is_split_into_words=True, 
#                                             padding='max_length', max_length=512, truncation=True, return_length=True)

#     batch_tags = []
#     length = []
#     for sample_i, tags in enumerate(examples["tags"]):
#         wids = tokenized.word_ids(sample_i)
#         real_tags = []
#         for wid in wids:
#             if wid is None:
#                 real_tags.append(-100)  #Ignore index for pytorch
#             else:
#                 real_tags.append(tags[wid])
#         batch_tags.append(real_tags)
#         length.append(sum(tokenized["attention_mask"][sample_i]))
    
#     return {
#         "input_ids" : tokenized["input_ids"],
#         "attention_mask" : tokenized["attention_mask"],
#         "length" : length,
#         "tags" : batch_tags
#     }

def dataset_map_raw2prompt(tokenizer, tag_names, examples):
    tokenized = tokenizer.batch_encode_plus(examples["tokens"], add_special_tokens=True, is_split_into_words=True, 
                                            padding='max_length', max_length=512, truncation=True, return_length=True)

    prompts = copy.deepcopy(tokenized["input_ids"])
    #batch_encode_plus is fast

    length = []
    mapped_poses = []
    mapped_tags = []
    for sample_i, tags in enumerate(examples["tags"]):
        wids = tokenized.word_ids(sample_i)
       # texts = tokenized["input_ids"][sample_i]
        real_len = sum(tokenized["attention_mask"][sample_i])
        mapped_pos = []
        mapped_tag = []
        j = 0
       # pdb.set_trace()
        for wid in wids:
            if wid != None:
                tag = tag_names[tags[wid]]
                if tag != 'O':
                    mapped_pos.append(j)    #record where to map to labelword
                    mapped_tag.append(tag)  #record the labelword
            j += 1
       # pdb.set_trace()
       # batch_tags.append(real_tags)
        length.append(real_len)
        mapped_poses.append(mapped_pos)
        mapped_tags.append(mapped_tag)
    
    tokenized3 = tokenizer.batch_encode_plus(mapped_tags, add_special_tokens=False, is_split_into_words=True, padding=False, truncation=False,
                                                        return_attention_mask=False, return_token_type_ids=False)
    #pdb.set_trace()

    for sample_i, mapped in enumerate(mapped_poses):
        for j, pos in enumerate(mapped):
            prompts[sample_i][pos] = tokenized3["input_ids"][sample_i][j]
    
  #  pdb.set_trace()
    return {
        "input_ids" : tokenized["input_ids"],
       # "attention_mask" : tokenized["attention_mask"],
       # "length" : length,
        "tags" : prompts
    }

    
class NERDatasetsConfigs:
    with open(f"{get_base_dirname()}/assets/ner_datasets_configs.json") as f:
        configs = json.load(f)

def auto_get_tag_names(config : Configs):
   return NERDatasetsConfigs.configs[config.dataset_name]["tag_names"]

def auto_get_dataset(config : Configs):
    if config.dataset_name == "conll2003":
        return get_datasets("conll2003-base")
    elif config.dataset_name == "ontonotes5":
        return get_datasets("ontonotes5-base")
    else:
        raise RuntimeError(f"Can't get dataset {config.dataset_name}")

def auto_get_bert_name_or_path(config : Configs):
    bert = config.bert_name_or_path
    if not bert:
        bert = NERDatasetsConfigs.configs[config.dataset_name]["default_bert"]
    if os.path.exists(f"{get_base_dirname()}/assets/pretrained_models/{bert}"):
        return f"{get_base_dirname()}/assets/pretrained_models/{bert}"
    else:
        return bert
        
def auto_get_tokenizer(config : Configs):
    if not config.using_bert:
        return NERTokenizerFromDataset.from_pretrained(f"{get_base_dirname()}/assets/pretrained_models/tokenizer-{config.dataset_name}")
    else:
        return AutoTokenizer.from_pretrained(auto_get_bert_name_or_path(config))

def auto_create_model(config : Configs, tokenizer):
    if config.model_name == "BiLSTM-Linear":
        return NER_BiLSTM_Linear(tokenizer.vocab_size, len(auto_get_tag_names(config)), 
                    config.embedding_size, config.lstm_layers, config.lstm_hidden_size, config.dropout_ratio)
    elif config.model_name == "BiLSTM-Linear-CRF":
        return NER_BiLSTM_Linear_CRF(tokenizer.vocab_size, len(auto_get_tag_names(config)),
                    config.embedding_size, config.lstm_layers, config.lstm_hidden_size, config.dropout_ratio)
    elif config.model_name == "BERT-Linear":
        return NER_BERT_Linear(auto_get_bert_name_or_path(config), len(auto_get_tag_names(config)), config.dropout_ratio)
    elif config.model_name == "BERT-Linear-CRF":
        return NER_BERT_Linear_CRF(auto_get_bert_name_or_path(config), len(auto_get_tag_names(config)), config.dropout_ratio)
    elif config.model_name == "BERT-BiLSTM-Linear":
        return NER_BERT_BiLSTM_Linear(auto_get_bert_name_or_path(config), len(auto_get_tag_names(config)), config.lstm_layers,
                            config.lstm_hidden_size, config.dropout_ratio)
    elif config.model_name == "BERT-BiLSTM-Linear-CRF":
        return NER_BERT_BiLSTM_Linear_CRF(auto_get_bert_name_or_path(config), len(auto_get_tag_names(config)), config.lstm_layers, config.lstm_hidden_size,
                            config.dropout_ratio)
    elif config.model_name == "BERT-Prompt":
        return NER_BERT_Prompt(f"{auto_get_bert_name_or_path(config)}-{config.dataset_name}-prompt", auto_get_tag_names(config))
    else:
        raise RuntimeError(f"Can't get model {config}")
