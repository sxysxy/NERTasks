#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# 实现Template-free prompt ner 的一些工作

from lib2to3.pgen2 import token
from myutils import Configs, auto_get_bert_name_or_path, auto_get_dataset, auto_get_tag_names
from transformers import BertForMaskedLM, BertTokenizer
import numpy as np
import torch

if __name__ == "__main__":
    config = Configs.parse_from_argv()
    if config.model_name != "BERT(Prompt)":
        print(f"Ignored model_name = {config.model_name}")
    
    tag_names = auto_get_tag_names(config)
    tag_name_map = dict()
    for tag in tag_names:
        tag_name_map[tag] = set()

    train_dataset = auto_get_dataset(config)["train"]
    for text, tag_ in zip(train_dataset["tokens"], train_dataset["tags"]):
        tag = [tag_names[t] for t in tag_]
        for c, t in zip(text, tag):
            tag_name_map[t].add(c)
    
    bert : BertForMaskedLM = BertForMaskedLM.from_pretrained(auto_get_bert_name_or_path(config))
    tokenizer : BertTokenizer = BertTokenizer.from_pretrained(auto_get_bert_name_or_path(config))
    bert_embed = bert.get_input_embeddings().weight
    tag_embeds_map = dict()

    for tag in tag_name_map:
        if tag == 'O':
            continue
        virtual_embeds = []
       # virtual_token = f"[{tag}]"
        virtual_token = tag
        for w in tag_name_map[tag]:
            ids = tokenizer(w, add_special_tokens=False, is_split_into_words=True, return_attention_mask=False, return_token_type_ids=False)['input_ids']
            if len(ids) != 1:
                continue
            virtual_embeds.append(bert_embed[ids[0]].unsqueeze(0))
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

    bert.save_pretrained(auto_get_bert_name_or_path(config) + f"-{config.dataset_name}-prompt")
    tokenizer.save_pretrained(auto_get_bert_name_or_path(config) + f"-{config.dataset_name}-prompt")
    
    




            
    
 