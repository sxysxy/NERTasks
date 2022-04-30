#-*- coding: utf-8
# Author: 石响宇(18281273@bjtu.edu.cn)
# License: LGPL-v3
# 加载数据集，使用这个文件的时候，要确保在datasets.load_dataset(本文件)之前设置os.environ["raw_datasets_path"]，数据集的路径
# 看起来可能很像TextMosaic里面的代码，因为TextMosaic里那套代码也是我写的

import datasets
import os
import numpy as np
import random
import ujson as json

assets_path = os.environ["assets_path"]
with open(f"{assets_path}/ner_datasets_configs.json") as f:
    ner_datasets_configs = json.load(f)

class DatasetConfig(datasets.BuilderConfig):
    def __init__(self, config, **kwargs):
        super(DatasetConfig, self).__init__(**kwargs)
        self.config = config

class AllDatasets(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0") 

    BUILDER_CONFIGS = [
        DatasetConfig(name = "conll2003-base", 
            config = {
                "tag_names" : ner_datasets_configs["conll2003"]["tag_names"],
                "train_url" : f"{assets_path}/raw_datasets/CoNLL2003_NER/train",
                "test_url" :  f"{assets_path}/raw_datasets/CoNLL2003_NER/test",
                "data_aug" : None
            }),
        DatasetConfig(name = "ontonotes5-base",
            config = {
                "tag_names" : ner_datasets_configs["ontonotes5"]["tag_names"],
                "train_url" : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.train.bmes",
                "test_url"  : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.test.bmes",
                "data_aug" : None
            }),
    ]

    def _info(self):
        if not self.config.name.startswith("MLM"):
            features = datasets.Features(
                {
                    "tokens" : datasets.Sequence( datasets.Value("string") ),
                    "tags" : datasets.Sequence( datasets.ClassLabel(names=self.config.config["tag_names"]))
                }
            ) 
            return datasets.DatasetInfo( 
                description="",
                features = features
            )
        else:
            features = datasets.Features(
                {
                    "text" : datasets.Sequence( datasets.Value("string") ),
                }
            ) 
            return datasets.DatasetInfo( 
                description="",
                features = features
            )

    def _split_generators(self, _):
        train_sp = datasets.SplitGenerator(name = datasets.Split.TRAIN, 
                gen_kwargs={
                    "split" : "train",
                    "filename" : self.config.config["train_url"]
                }
            )

        if not self.config.name.startswith("MLM"):
            test_sp = datasets.SplitGenerator(name = datasets.Split.TEST, 
                gen_kwargs={
                    "split" : "test",
                    "filename" : self.config.config["test_url"]
                }
            )
            return [train_sp, test_sp]
        else:
            return [train_sp]

    def _generate_examples(self, **args):
        if self.config.name.startswith("conll2003"):
            print("Dataset: Loading conll2003")
            with open(f"{args['filename']}/seq.in") as f:
                origin_texts = list(map(lambda x : x.strip().split(' '), filter(lambda t : len(t.strip()) > 0, f.readlines())))
            with open(f"{args['filename']}/seq.out") as f:
                origin_tags = list(map(lambda x : x.strip().split(' '), filter(lambda t : len(t.strip()) > 0, f.readlines())))
            s = 0
            for pair in zip(origin_texts, origin_tags):
                yield s, {
                    "tokens" : pair[0],
                    "tags" : pair[1]
                }
                s += len(pair[0])
        else:
            pass