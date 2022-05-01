#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn)
# License: LGPL-v3
# 单元测试

from myutils import NERTokenizerFromDataset, auto_get_tag_names, dataset_map_raw2ner, NERDatasetsConfigs, get_base_dirname, get_datasets, dataset_map_raw2prompt
import pdb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer

def test_tokenizer():
    tokenizer = NERTokenizerFromDataset()
    tokenizer.rebuild_vocab([
        list("自然语言处理"),
        ["(", "Natural", "Language", "Processing", ",", "NLP", ")"],
        list("是计算机科学领域与人工智能领域中的一个重要方向。"),
        list("它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。"),
        list("自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"),
        list("因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。"),
        list("自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。因而它是计算机科学的一部分。")
    ])
    batch_encoded = tokenizer.batch_encode_plus([
        list("自然语言处理"),
        ["Natural", "Language", "Processing"],
        list("理论和方法")
    ], max_length=10)

    pdb.set_trace()

def test_loaddataset1():
    raw_dataset = get_datasets('conll2003-base')
    tokenizer : BertTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    sample_raw = raw_dataset["train"][0]

   # ner_dataset = raw_dataset.map(lambda x : dataset_map_raw2ner(tokenizer, x), batched=True)
   # ner_dataset.set_format('torch', columns=["input_ids", "attention_mask", "length", "tags"])
    ner_dataset, _ = dataset_map_raw2ner(raw_dataset, tokenizer)
    sample_ner = ner_dataset["train"][0]

    loader = DataLoader(ner_dataset["train"], batch_size=1)

    sample_batch = loader.__iter__().next()

    pdb.set_trace()

def test_loaddataset2():
    raw_dataset = get_datasets('ontonotes5-base')
    tokenizer : BertTokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    sample_raw = raw_dataset["train"][0]
    pdb.set_trace()

   # ner_dataset = raw_dataset.map(lambda x : dataset_map_raw2ner(tokenizer, x), batched=True)
   # ner_dataset.set_format('torch', columns=["input_ids", "attention_mask", "length", "tags"])
    ner_dataset, _ = dataset_map_raw2ner(raw_dataset, tokenizer)
    sample_ner = ner_dataset["train"][0]

    loader = DataLoader(ner_dataset["train"], batch_size=1)

    sample_batch = loader.__iter__().next()

    pdb.set_trace()

def test_loaddataset_prompt():
    raw_dataset = get_datasets('conll2003-base')
    tokenizer : BertTokenizer = AutoTokenizer.from_pretrained(f"{get_base_dirname()}/assets/pretrained_models/bert-base-uncased-conll2003-prompt")
    tag_names = NERDatasetsConfigs.configs["conll2003"]["tag_names"]

    prompt_dataset = raw_dataset.map(lambda x : dataset_map_raw2prompt(tokenizer, tag_names, x), batched=True)
    prompt_dataset.set_format('torch', columns=["input_ids", "attention_mask", "length", "tags"])

    sample = prompt_dataset["train"][0]

    loader = DataLoader(prompt_dataset["train"], batch_size=4)
    sample_batch = loader.__iter__().next()

    pdb.set_trace()



if __name__ == "__main__":
  #  test_tokenizer()
    test_loaddataset2()
  #  test_loaddataset_prompt()
