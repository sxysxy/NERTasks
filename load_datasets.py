#-*- coding: utf-8
# Author: 石响宇(18281273@bjtu.edu.cn)
# License: LGPL-v3
# 加载数据集，使用这个文件的时候，要确保在datasets.load_dataset(本文件)之前设置os.environ["raw_datasets_path"]，数据集的路径
# 看起来可能很像TextMosaic里面的代码，因为TextMosaic里那套代码也是我写的

from distutils.command.config import config
from fileinput import filename
import datasets
import os
import numpy as np
import random
import ujson as json
import re
import pdb
import codecs 

assets_path = os.environ["assets_path"]
with open(f"{assets_path}/ner_datasets_configs.json") as f:
    ner_datasets_configs = json.load(f)

def map_tagname_bio2io(tag_names):
    ios = set()
    for tag in tag_names:
        ios.add(f"I-{tag[2:]}" if tag != 'O' else tag)
    return list(ios) 

def convert_tagged_bio2io(tags):
    iotags = []
    for tag in tags:
        iotag = []
        for j in range(len(tag)):
            if tag[j] == 'O':
                iotag.append('O')
            else:
                iotag.append(f"I-{tag[j][2:]}")
        iotags.append(iotag)
    return iotags



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
            }),
        DatasetConfig(name = "ontonotes5-base",
            config = {
                "tag_names" : ner_datasets_configs["ontonotes5"]["tag_names"],
                "train_url" : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.train.bmes",
                "test_url"  : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.test.bmes",
            }),
        DatasetConfig(name = "ccks2019-base",
            config = {
                "tag_names" : ner_datasets_configs["ccks2019"]["tag_names"],
                "train_url" : f"{assets_path}/raw_datasets/CCKS2019",
                "test_url" : f"{assets_path}/raw_datasets/CCKS2019"
            }),
        DatasetConfig(name = "conll2003-io", 
            config = {
                "tag_names" : map_tagname_bio2io( ner_datasets_configs["conll2003"]["tag_names"] ),
                "train_url" : f"{assets_path}/raw_datasets/CoNLL2003_NER/train",
                "test_url" :  f"{assets_path}/raw_datasets/CoNLL2003_NER/test",
            }),
        DatasetConfig(name = "ontonotes5-io",
            config = {
                "tag_names" : map_tagname_bio2io( ner_datasets_configs["ontonotes5"]["tag_names"] ),
                "train_url" : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.train.bmes",
                "test_url"  : f"{assets_path}/raw_datasets/ontonotes5_ch_ner/ontonotes5.test.bmes",
            }),
        DatasetConfig(name = "ccks2019-io",
            config = {
                "tag_names" : map_tagname_bio2io( ner_datasets_configs["ccks2019"]["tag_names"] ),
                "train_url" : f"{assets_path}/raw_datasets/CCKS2019",
                "test_url" : f"{assets_path}/raw_datasets/CCKS2019"
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
        origin_texts = []
        origin_tags = []
        if self.config.name.startswith("conll2003"):
            print("Dataset: Loading conll2003")
            with open(f"{args['filename']}/seq.in") as f:
                origin_texts = list(map(lambda x : x.strip().split(' '), filter(lambda t : len(t.strip()) > 0, f.readlines())))
            with open(f"{args['filename']}/seq.out") as f:
                origin_tags = list(map(lambda x : x.strip().split(' '), filter(lambda t : len(t.strip()) > 0, f.readlines())))
            if self.config.name.endswith('io'):
                origin_tags = convert_tagged_bio2io(origin_tags)
        elif self.config.name.startswith("ontonotes5"):
            print("Dataset: Loading OntoNotes5")
            with open(args["filename"], "r") as f:
                origin_texts = []
                origin_tags = []
                text = []
                tag = []
                for line in f.readlines():  
                    a = line.strip().split(' ')
                    if len(a) != 2:
                        if len(text) > 0:
                            origin_texts.append(text)
                            origin_tags.append(tag)
                        text = []
                        tag = []
                    else:
                        text.append(a[0])
                        tag.append(a[1])
                if len(text) > 0:
                    origin_texts.append(text)
                    origin_tags.append(tag)
        elif self.config.name.startswith("ccks2019"):
            print("Dataset: Loading CCKS2019")
            def read_ccks_jsonl(filename, scheme='BIO'):
                texts = []
                tags = []
                with codecs.open(filename, "r", 'utf_8_sig') as f:
                    for line in f.readlines():
                        if len(line.strip()) == 0:
                            continue
                        sample = json.loads(line)
                        t = list(sample["originalText"])
                        texts.append(t)
                        len_t = len(t)
                        tag = ['O'] * len_t
                        for ent in sample["entities"]:
                            if ent["start_pos"] >= 510 or ent["end_pos"] > 510:
                                continue
                            spos = ent["start_pos"]
                            epos = ent["end_pos"]
                            ent_type = ent["label_type"]
                            if scheme == "BIO":
                                tag[spos] = f"B-{ent_type}"
                                for p in range(spos+1, epos):
                                    tag[p] = f"I-{ent_type}"
                            elif scheme == "IO":
                                for p in range(spos, epos):
                                    tag[p] = f"I-{ent_type}"
                            else:
                                raise RuntimeError(f"Unknown scheme {scheme}")
                        tags.append(tag)
                return texts, tags
            if args["split"] == "train":
                origin_texts, origin_tags = read_ccks_jsonl(os.path.join(args["filename"], "subtask1_training_part1.txt"), "IO" if self.config.name.endswith('io') else "BIO")
                a, b = read_ccks_jsonl(os.path.join(args["filename"], "subtask1_training_part2.txt"), "IO" if self.config.name.endswith('io') else "BIO")
                origin_texts.extend(a)
                origin_tags.extend(b)
            else:
                origin_texts, origin_tags = read_ccks_jsonl(os.path.join(args["filename"], "subtask1_test_set_with_answer.json"), "IO" if self.config.name.endswith('io') else "BIO")
        else:
            raise RuntimeError(f"Unresolved dataset {self.config.name}")

        s = 0
        for pair in zip(origin_texts, origin_tags):
            yield s, {
                "tokens" : pair[0],
                "tags" : pair[1]
            }
            s += len(pair[0])