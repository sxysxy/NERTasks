#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# 实现Template-free prompt ner 的一些工作

from lib2to3.pgen2 import token
from myutils import Configs, auto_get_bert_name_or_path, auto_get_dataset, auto_get_tag_names, dataset_map_raw2ner, get_base_dirname
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np
import torch
import os

MAX_VIRTUAL_AVERAGE_COUNT = 5

class WordWithFreq:
    def __init__(self, w) -> None:
        self.word = w
        self.count = 1
    
    def __hash__(self):
        return hash(self.word)

    def __eq__(self, __o) -> bool:
        if isinstance(__o, WordWithFreq):
            return self.word == __o.word
        elif isinstance(__o, str):
            return self.word == __o

    def __lt__(self, o):
        return self.count > o.count

    def add(self):
        self.count += 1

if __name__ == "__main__":
    config = Configs.parse_from_argv()
    if config.model_name != "BERT-Prompt":
        print(f"Ignored model_name = {config.model_name}")
    
    tag_names = auto_get_tag_names(config)
    tag_name_map = dict()

    train_dataset = auto_get_dataset(config)["train"]
    for text, tag_ in zip(train_dataset["tokens"], train_dataset["tags"]):
        tag = [tag_names[t] for t in tag_]
        for c, t in zip(text, tag):
            if t == 'O':
                continue
           # t2 = t[2:]
            t2 = t
            if t2 in tag_name_map:
                if c in tag_name_map[t2]:
                    tag_name_map[t2][c] += 1
                else:
                    tag_name_map[t2][c] = 1
            else:
                tag_name_map[t2] = {c : 1}
    
    data_search_results = {}
    for tag in tag_name_map:
        x = list(zip(tag_name_map[tag].values(), tag_name_map[tag].keys()))
        x = sorted(x, reverse=True)
        data_search_results[tag] = x
    
    model_name = config.bert_name_or_path
    if os.path.exists(f"{get_base_dirname()}/assets/pretrained_models/{model_name}"):
        model_name = f"{get_base_dirname()}/assets/pretrained_models/{model_name}"
        
    bert : BertForMaskedLM = BertForMaskedLM.from_pretrained(model_name)
    tokenizer : BertTokenizer = BertTokenizer.from_pretrained(model_name)


    bert_embed = bert.get_input_embeddings().weight
    tag_embeds_map = dict()
    special_tokens_ids = set([tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.unk_token_id, tokenizer.pad_token_id])

    for tag in tag_name_map:
        if tag == 'O':
            continue
        virtual_embeds = []
       # virtual_token = f"[{tag}]"
        virtual_token = tag

        for cnt, w in data_search_results[tag]:
            ids = tokenizer(w, add_special_tokens=False, is_split_into_words=True, return_attention_mask=False, return_token_type_ids=False)['input_ids']
            if len(ids) != 1:
                continue
            virtual_embeds.append(bert_embed[ids[0]].unsqueeze(0))
            if len(virtual_embeds) > MAX_VIRTUAL_AVERAGE_COUNT:
                break
        '''
        for w in tag_name_map[tag]:
            ids = tokenizer(w, add_special_tokens=False, is_split_into_words=True, return_attention_mask=False, return_token_type_ids=False)['input_ids']
            if len(ids) != 1:
                continue
            if ids[0] in special_tokens_ids:
                continue
            virtual_embeds.append(bert_embed[ids[0]].unsqueeze(0))
            if len(virtual_embeds) > MAX_VIRTUAL_AVERAGE_COUNT:
                break
        '''
        if len(virtual_embeds) == 0:
            raise RuntimeError(f"Can't generate virtual word for {tag}")
        virtual_embeds = torch.cat(virtual_embeds, dim=0)
        virtual_embed = torch.mean(virtual_embeds, dim=0).unsqueeze(0)
        tag_embeds_map[virtual_token] = virtual_embed

    #new_embeds = []
    #for tag in tag_embeds_map:
    #    embd = tag_embeds_map[tag]
    #    tokenizer.add_tokens(new_tokens=[], special_tokens=tag)
    #    new_embeds.append(embd)
    new_tags = []
    new_embeds = []
    for tag in tag_embeds_map:
        new_tags.append(tag)
        new_embeds.append(tag_embeds_map[tag])
        
    num_added = tokenizer.add_special_tokens({"additional_special_tokens" : new_tags})
    #num_added = tokenizer.add_tokens(new_tokens=[],special_tokens=new_tags)
    if num_added != len(new_tags):
        print("Add failed")
        exit(1)
    new_input_embeddings = torch.cat([bert_embed] + new_embeds, dim=0)
    
    bert.resize_token_embeddings(len(tokenizer))
    bert.set_input_embeddings(torch.nn.Embedding.from_pretrained(new_input_embeddings))

    save_model_path = f"{get_base_dirname()}/assets/pretrained_models/{os.path.split( config.bert_name_or_path )[-1]}-{config.dataset_name}-prompt"
    bert.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)





            
    
 