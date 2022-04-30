#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: LGPL-v3
# 训练器

from typing import Callable
from numpy import full
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    //thanks to https://blog.csdn.net/ltochange/article/details/116524264
    Warmup预热学习率：先从一个较小的学习率线性增加至原来设置的学习率，再进行学习率的线性衰减
   
    当 current_step < num_warmup_steps时，
    new_lr =current_step/num_warmup_steps * base_lr
    当current_step >= num_warmup_steps时，
    new_lr =(num_training_steps - current_step) / (num_training_steps -num_warmup_steps) * base_lr

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
         # 自定义函数
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class NERTrainer:
    '''
    model : 模型
    optimizer : 优化器
    warmup_ratio : 学习率预热轮数的比率。None不使用学习率预热。
    label_smooth_factor : 标签平滑因子。None不使用标签平滑。
    clip_grad_norm : 梯度裁剪。None则不使用梯度裁剪。
    grad_acc : 梯度累积数。如果不梯度累积就设为1。
    '''
    def __init__(self, 
            model : nn.Module, 
            optimizer : optim.Optimizer,
            warmup_ratio : float,
            label_smooth_factor : float,
            clip_grad_norm : float,
            grad_acc : int,
            data_columns : list,
                ):
        self.model = model
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.label_smooth_factor = label_smooth_factor
        self.label_smoother = None
        if label_smooth_factor:
            self.label_smoother = LabelSmoother(label_smooth_factor, ignore_index=-100)
        self.clip_grad_norm = clip_grad_norm
        self.grad_acc = grad_acc if grad_acc else 1
        self.data_columns = data_columns

    def train(self, num_epochs, train_dataloader : DataLoader, eval_function : Callable):
        '''
        num_epochs : 训练轮数
        train_dataloader : 训练集数据加载器。
        eval_function(model) : 评测模型的函数，应当返回一个dict，字段随便。如果为None那就没有评测。
        '''
        lr_sched = None
        if self.warmup_ratio:
            lr_sched = get_linear_schedule_with_warmup(self.optimizer, min(1, int(num_epochs * self.warmup_ratio)), num_epochs)
        optimizer = self.optimizer
        model = self.model
        model.train()
        optimizer.zero_grad()
        
        all_metrics = {
            "metrics_each_epoch" : []
        }

        time_used_sec = 0
        for epoch_i in range(num_epochs):
            def optim_step():
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            with tqdm(desc=f"Epoch {epoch_i + 1}", total=len(train_dataloader), unit="ba") as bar:
                epoch_loss = 0
                grad_acc_cnt = 0
                start_time = datetime.now()  
                for batch_i, batch in enumerate(train_dataloader):
                    batch_gpu = {}
                    for col in self.data_columns:
                        batch_gpu[col] = batch[col].cuda()
                    # if self.label_smoother is None:
                    loss : torch.Tensor = model(**batch_gpu)['loss'].mean() / self.grad_acc
                    #else:
                    #   loss : torch.Tensor = self.label_smoother(model(**batch_gpu), batch_gpu['tags']) / self.grad_acc
                    loss.backward()
                    epoch_loss += loss.item()
                    grad_acc_cnt += 1
                    if grad_acc_cnt % self.grad_acc == 0:
                        optim_step()
                        grad_acc_cnt = 0
                    bar.update(1)
                if grad_acc_cnt != 0:
                    optim_step()
                if lr_sched:
                    lr_sched.step()
                bar.update(1)

                end_time = datetime.now()
                dur = end_time - start_time
                epoch_time = dur.seconds + 1e-6 * dur.microseconds
            
            metrics_ep = {}
            metrics_ep["loss"] = epoch_loss
            metrics_ep['time_sec'] = epoch_time

            if eval_function:
                model.eval()
                metrics_ep = eval_function(metrics_ep)
                model.train()
        
            all_metrics["metrics_each_epoch"].append(metrics_ep)
        
        all_metrics["num_epoches"] = num_epochs
        for m in all_metrics["metrics_each_epoch"]:
            time_used_sec += m["time_sec"]
        all_metrics["time_sec"] = time_used_sec
        
        return all_metrics
        