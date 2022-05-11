#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# main


import random
from matplotlib.cbook import ls_mapper
from transformers import set_seed
from myutils import (Configs, 
    auto_create_model, auto_get_dataset, auto_get_tag_names, 
    auto_get_tokenizer, dataset_map_raw2ner, dataset_map_raw2prompt, get_base_dirname, 
    get_ner_evaluation
)
from mytrainer import NERTrainer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from datetime import datetime
import ujson as json
from ner_models import INERModel

def main():
    config = Configs.parse_from_argv()

    set_seed(config.random_seed)

    raw_dataset = auto_get_dataset(config)
    
    if config.few_shot:
        raw_dataset["train"] = raw_dataset["train"].shuffle(config.few_shot_seed).select(list(range(int(len(raw_dataset["train"]) * config.few_shot))))
    
    tokenizer = auto_get_tokenizer(config)
    model : INERModel = auto_create_model(config, tokenizer).cuda()
    if not config.using_prompt:
        ner_dataset, columns = dataset_map_raw2ner(raw_dataset, tokenizer)
    else:
        ner_dataset, columns = dataset_map_raw2prompt(raw_dataset, tokenizer, auto_get_tag_names(config))

    #optimizer = optim.Adam(model.parameters(), lr=config.ner_lr)
    optimizer = optim.AdamW(model.parameters(), lr=config.ner_lr, weight_decay=config.ner_weight_decay)
    trainer = NERTrainer(model, 
                        optimizer=optimizer, 
                        warmup_ratio=config.warmup_ratio,
                        label_smooth_factor=config.label_smooth_factor,
                        clip_grad_norm=config.clip_grad_norm,
                        grad_acc=config.grad_acc, 
                        data_columns=columns)
    tag_names = auto_get_tag_names(config)
    metric_eval = get_ner_evaluation()
    
    test_loader = DataLoader(ner_dataset["test"], batch_size=config.batch_size, pin_memory=True)
    def eval_function(metrics_data : dict):
        with torch.no_grad():
            y_pred = []
            y_true = []
            for batch in test_loader:
                batch_gpu = {}
                for col in columns:
                    batch_gpu[col] = batch[col].cuda()
                decoded = model.decode(**batch_gpu)     #[batch_size, seq_length]
                y_pred.append(decoded)
                y_true.append(batch["tags"].view(-1)[1:-1])

            if isinstance(y_pred[0], torch.Tensor):
                y_pred = [p.contiguous().view(-1) for p in y_pred]
            elif isinstance(y_pred[0], list):
                y_pred = [sum(p, []) for p in y_pred]
              #  y_pred = np.concatenate(y_pred)
            else:
                raise RuntimeError("Unkown decoded type")

           # y_true = torch.cat(y_true).detach().cpu()

            true_predictions = [
                [tag_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(y_pred, y_true)
            ]
            true_labels = [
                [tag_names[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(y_pred, y_true)
            ]
            results = metric_eval.compute(predictions=true_predictions, references=true_labels)
            if config.f1 == Configs.OVERALL_MICRO:
                metrics_data.update({
                    "f1": results["overall_micro_f1"],
                })
            elif config.f1 == Configs.OVERALL_MACRO:
                metrics_data.update({
                    "f1": results["overall_macro_f1"],
                })
            metrics_data.update({
                "accuracy": results["overall_accuracy"]
                })
            print(metrics_data)
            return metrics_data
        
    class EvalFunctionHook:
        def __init__(self) -> None:
            self.best_f1 = 0
        def __call__(self, metric_data : dict):
            m = eval_function(metric_data)
            if not config.save_model:
                return m
            f1 = m["f1"]
            if f1 > self.best_f1:
                savep = f"{get_base_dirname()}/results/{config.dataset_name}-{config.model_name}"
                os.makedirs(savep, mode=0o755, exist_ok=True)
                tokenizer.save_pretrained(savep)
                torch.save(model, os.path.join(savep, "model.bin"))
                self.best_f1 = f1
            return m
            
    evaluator = EvalFunctionHook()
        
    all_metrics = trainer.train(config.ner_epoches, DataLoader(ner_dataset["train"], batch_size=config.batch_size, pin_memory=True), evaluator)
    all_metrics["config"] = config.__dict__
    best_f1 = 0
    for m in all_metrics["metrics_each_epoch"]:
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
    all_metrics["best_f1"] = best_f1
    all_metrics["GPU"] = torch.cuda.get_device_name()
    os.makedirs("results", exist_ok=True, mode=0o755)

    now = datetime.now()
    with open(f"results/{config.dataset_name}-{config.model_name}-{str(now.date())}-{now.hour}-{now.minute}-{now.second}.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    

if __name__ == "__main__":
    main()
