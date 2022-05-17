from myutils import Configs, auto_get_bert_name_or_path, auto_get_dataset, auto_get_tag_names, auto_get_tokenizer, dataset_map_raw2ner, get_base_dirname, NERDatasetsConfigs, get_datasets
from transformers import BertForMaskedLM, BertTokenizer
import torch
import os

if __name__ == "__main__":
    config = Configs.parse_from_argv()
    train_dataset = get_datasets(f"{config.dataset_name}-base")["train"]
    if config.few_shot:
        train_dataset = train_dataset.shuffle(config.few_shot_seed).select(list(range(int(len(train_dataset) * config.few_shot))))
    train_dataset = dataset_map_raw2ner(train_dataset, auto_get_tokenizer(config))[0]

    tags = auto_get_tag_names(config)
    starttagid = len(tags)
    endtagid = starttagid+1
    tag_size = len(tags)+2

    transition = torch.zeros(tag_size, tag_size)
    num_trans = 0
    for sent_tag in train_dataset["tags"]:
        sent_tag = sent_tag[1:-1]
        transition[starttagid, sent_tag[0]] += 1
        num_trans += 1
        for i in range(len(sent_tag)-1):
            transition[sent_tag[i], sent_tag[i+1]] += 1
            num_trans += 1
        transition[sent_tag[-1], endtagid] += 1
        num_trans += 1
    #transition = torch.log((transition + 1) / num_trans).transpose(0,1)
    transition = transition.transpose(0,1) / num_trans

    assert torch.isnan(transition).int().sum() == 0  #no nan

    save_dir = f"{get_base_dirname()}/assets/pretrained_transitions"
    os.makedirs(save_dir, mode=0o755, exist_ok=True)
    if config.few_shot:
        few_name = config.dataset_name + "-" + str(config.few_shot) + "-" + str(config.few_shot_seed) 
        torch.save(transition, f"{save_dir}/{few_name}-transition.bin")
    else:
        torch.save(transition, f"{save_dir}/{config.dataset_name}-transition.bin")
    