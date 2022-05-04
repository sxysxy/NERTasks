import os
import sys

label_smooth_factor = -1
#ner_epoches = 12
#ner_epoches = 20
ner_epoches = 30
batch_size = 1
grad_acc = 32
lstm_layers = 2
lstm_hidden_size = 256
few_shot = 0.01
few_shot_seed = 11451419198
#bert = "medbert-base-chinese"
#bert = 'bert-base-chinese'
#few_shot = -1
#bert = None #Auto
bert = 'biobert-base-cased-v1.2'
#bert = 'bert-base-uncased'


argv = sys.argv[1:]

if len(argv) % 3 != 0:
    print("Useage: easy_to_run {dataset model lr}*")

n_group = len(argv) // 3
for i in range(n_group):
    args = argv[i * 3 : (i+1) * 3]
    cmd = f"python main.py --dataset {args[0]} --model_name {args[1]} --ner_lr {args[2]} --label_smooth_factor {label_smooth_factor} --ner_epoches {ner_epoches} --batch_size {batch_size} --grad_acc {grad_acc} --lstm_layers {lstm_layers} --lstm_hidden_size {lstm_hidden_size} --few_shot {few_shot} {('--bert_name_or_path %s' % bert) if bert else ''} --few_shot_seed {few_shot_seed}"
    os.system(cmd)    

