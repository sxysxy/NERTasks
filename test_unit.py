#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn)
# License: GPL-v3
# 单元测试

from myutils import NERTokenizerFromDataset, load_datasets
import pdb
from torch.utils.data import DataLoader
from transformers import BertTokenizer

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

def test_loaddataset():
    dataset = load_datasets('conll2003-base')
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    ds_train = dataset["train"]
    ds_test = dataset["test"]

    sample = ds_train[0]

    loader = DataLoader(ds_train, batch_size=4)

    def map_rawds(examples):
        pass
        

    sample_batch = loader.__iter__().next()

    pdb.set_trace()


if __name__ == "__main__":
    pdb.set_trace()