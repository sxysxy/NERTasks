#-*- coding:utf-8
# Author: 石响宇(18281273@bjtu.edu.cn) 
# License: GPL-v3
# NER模型（们）
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils import Configs

class INERModel(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
    
    def forward(self, X):
        return {
            "logits" : torch.tensor(0).to(self.device),
            "loss" : torch.tensor(0).to(self.device)
        }

    
class NER_BiLSTM_Linear(INERModel):
    def __init__(self, device, vocab_size, tag_size) -> None:
        super().__init__(device)

