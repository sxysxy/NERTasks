#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# 通过数据集构建tokenizer，类型为NERTokenizerFromDataset，接口类似于BertTokenizer

import os
from myutils import NERDatasetsConfigs, get_base_dirname, NERTokenizerFromDataset, load_datasets

if __name__ == "__main__":
    pretrain_dir = f"{get_base_dirname()}/assets/pretrained_models"
    os.makedirs(pretrain_dir, exist_ok=True, mode=0o755)

    for ds_name in NERDatasetsConfigs.configs.keys():
        trainset = load_datasets(f"{ds_name}-base")["train"]
        tokenizer = NERTokenizerFromDataset()
        tokenizer.rebuild_vocab(trainset["tokens"])
        tokenizer.save_pretrained(f"{pretrain_dir}/tokenizer-{ds_name}")
        
        


