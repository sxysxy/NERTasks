# NERTasks

# Contents
## [What's It?](#0)
## [1. Requrements](#1)
## [2. Install Dependencies](#2)
## [3. How To Prepare Datasets](#3)
## [4. Experiments](#4)
### [&nbsp; 4.1 Hyper Parameters](#4_1)
### [&nbsp; 4.2 Model Parameters](#4_2)
### [&nbsp; 4.3 Results](#4_3)
#### [&nbsp; &nbsp; 4.3.1 Full Data Results](#4_3_1)
#### [&nbsp; &nbsp; 4.3.2 Few Shot Results](#4_3_2)
## [5. Acknowledgement And Citations](#5)
### [&nbsp; 5.1 People And Orgnizations](#5_1)
### [&nbsp; 5.2 Third-Party Libraries](#5_2)

<p id="0"> </p>

# What's It? 

<h4><b> A simple NER framework. </b> </h4>

It implements:

<table>
<thead>
<tr><td></td><td>Item</td><td>Source/Reference</td></tr>
</thred>
<tbody>
<tr><td rowspan="8">Models</td></tr>
    <tr><td>BiLSTM-Linear</td><td><a href="https://www.researchgate.net/publication/13853244_Long_Short-term_Memory">Long Short-term Memory</a></td></tr>
    <tr><td>BiLSTM-Linear-CRF</td><td><a href="https://arxiv.org/abs/1603.01360">Neural Architectures for Named Entity Recognition</a></td></tr>
    <tr><td>BERT-Linear</td><td><a href="https://arxiv.org/abs/1810.04805">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a></td></tr>
    <tr><td>BERT-Linear-CRF</td><td></td></tr>
    <tr><td>BERT-BiLSTM-Linear</td><td></td></tr>
    <tr><td>BERT-BiLSTM-Linear-CRF</td><td></td></tr>
    <tr><td>BERT(Prompt)<br>EntLM Approach</td><td><a href="https://arxiv.org/abs/2109.13532">Template-free Prompt Tuning for Few-shot NER</a></td></tr>
<tr><td rowspan="4">Datasets</td></tr>
    <tr><td>CoNLL2003</td><td><a href="https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/CoNLL2003_NER">
yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification</a></td></tr>
    <tr><td>OntoNotes5</td><td><a href="https://catalog.ldc.upenn.edu/LDC2013T19">LDC2013T19</a></td></tr>
    <tr><td>CCKS2019 Subtask 1</td><td><a href="https://tianchi.aliyun.com/dataset/dataDetail?dataId=92085">TIANCHI</a></td></tr>
<tr><td rowspan="4">Traning Trick</td></tr>
    <tr><td>Gradient Accumulation</td><td></td></tr>
    <tr><td>Learning Rate Warmup</td><td></td></tr>
    <tr><td>Label Smooth</td><td></td></tr>
<tr><td rowspan="3">Misc</td></tr>
    <tr><td>Tokenizer from datasets</td><td></td></tr>
    <tr><td>NER Metrics</td><td><a href="https://github.com/chakki-works/seqeval">seqeval: A Python framework for sequence labeling evaluation</a></td></tr>
</tbody>
</table>

You can easily add your own models and datasets into this framework.

<p id="1"> </p>

# Requirements:

Linux(Tested)/Windows(Not Tested) with Nvidia GPUs.

<p id="2"> </p>

# Install Dependencies.

Recommend to use conda creating a python environment(python==3.9). For example:

```
conda create -n NER python=3.9
```

And run the bash script. If you are using windows, change its extname to .bat.

```
./install_dependencies.sh
```

<p id="3"> </p>

# How To Prepare Datasets

For some reason(copyright and some other things), I can't directly provide datasets to you. You should get the access to these datasets by yourself and put them in specified format into 'assert/raw_datasets' folder, see [here](assets/README.md).

<p id="4"> </p>

# Experiments

<p id="4_1"> </p>

## Hyper Parameters 

<table>
<thead>
<tr>
    <td>Optimizer</td>
    <td>Weight Decay</td> 
    <td>Warmup Ratio</td> 
    <td>Label Smoothing</td> 
    <td>Batch Size</td> 
    <td>Gradient Accumulation</td> 
    <td>Clip Grad Norm</td>
    <td>Total Epoches</td>
    <td>Random Seed</td>
    </tr>
</thead>
<tbody>
<tr>
    <td>AdamW</td>
    <td>5e-3</td>
    <td>0.2</td>
    <td>None</td>
    <td>1</td>
    <td>32</td>
    <td>1.0</td>
    <td>12</td>
    <td>233</td>
</tr>
</tbody>
</table>

Learning Rates:

<table>
<thead>
<tr><td> </td>
    <td>CoNLL2003</td>
    <td>OntoNotes5</td>
    <td>CCKS2019</td>
    </tr>
</thead>
<tbody>
<tr><td>BiLSTM-Linear</td></td>
    <td colspan="2" rowspan="2">0.001</td>
    <td rowspan="2">NA</td>
    </tr>
<tr><td>BiLSTM-Linear-CRF</td></tr>
<tr><td>BERT-Linear</td>
    <td colspan="3" rowspan="2">0.0001</td></tr>
<tr><td>BERT-Linear-CRF</td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td rowspan="3">0.0001</td>
    <td>3e-5</td>
    <td rowspan="2">NA</td>
    </tr>
    <tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td>1e-5</td></tr>
    <tr><td>BERT(Prompt)</td>
    <td>3e-5</td>
    <td>0.0001</td></tr>
</tbody>
</table>

<p id="4_2"> </p>

## Model Parameters

<table>
<thead>
<tr><td>BERT Model</td> <td>Embedding Size(For models without BERT)</td> <td> LSTM Hidden Size </td> <td> LSTM Layers </td> </tr>
</thead>
<tbody>
<tr><td> bert-base </td> <td> 256 </td> <td> 256 </td> <td> 2 </td></tr>
</tbody>
</table>

<p id="4_3"> </p>

## Results

<p id="4_3_1"></p>

### Full Data Results

<table>
<thead>
<tr>
    <td> Dataset </td>
    <td> Model </td>
    <td> Overall Span-Based Micro F1 </td>
    <td> Average Training Time Per Epoch<br>
    (On a Quadro RTX8000)</td>
</tr>
</thead>
<tbody>
<tr><td rowspan="7">CoNLL2003</td><td>BiLSTM-Linear</td>
    <td>0.6517005491858561</td>
    <td>13.98s</td></tr>
<tr><td>BiLSTM-Linear-CRF</td>
    <td>0.6949365863103882</td>
    <td>44.07s</td></tr>
<tr><td>BERT-Linear</td>
    <td>0.8983771483322356</td>
    <td>81.81s</td></tr>
<tr><td>BERT-Linear-CRF</td>
    <td>0.8977943835121128</td>
    <td>120.94s</td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td>0.8819152766110644</td>
    <td>117.37s</td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td>0.8873846891098599</td>
    <td>130.85s</td></tr>
<tr><td>BERT(Prompt)</td>
    <td>0.8803649678852272</td>
    <td>106.42s</td></tr>

<tr><td rowspan="7">OntoNotes5(Chinese)</td><td>BiLSTM-Linear</td>
    <td>0.637999350438454</td>
    <td>160.55s</td></tr>
<tr><td>BiLSTM-Linear-CRF</td>
    <td>0.7033358449208851</td>
    <td>319.87s</td></tr>
<tr><td>BERT-Linear</td>
    <td>0.7403041825095057</td>
    <td>413.20s</td></tr>
<tr><td>BERT-Linear-CRF</td>
    <td>0.7535838822161953</td>
    <td>595.71s</td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td>0.7511438739196745</td>
    <td>590.53s</td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td>0.7616389699353039</td>
    <td>800.23s</td></tr>
<tr><td>BERT(Prompt)</td>
    <td>0.7376454875023851</td>
    <td>485.56s</td></tr>

<tr><td rowspan="3">CCKS2019 Subtask1
    <td>BERT-Linear</td>
    <td>0.8057400574005741</td>
    <td>35.05s</td></tr>
    <td>BERT-Linear-CRF</td>
    <td>0.8119778310861113</td>
    <td>98.10s</td></tr>
    <td>BERT-Prompt</td>
    <td>0.7640559783943041</td>
    <td>39.71s</td></tr>

</tbody>
</thead>
</table>

<p id="4_3_2"></p>

### Few Shot Results

Few Shot test on CCKS2019:

I sampled 10 samples (total 1000) for training using fixed random seed. Here list the numbers of entities in few shot dataset:

```
{'手术': 9, '影像检查': 5, '疾病和诊断': 45, '解剖部位': 48, '实验室检验': 19, '药物': 10}
```

In this case, bert_name_or_path = trueto/medbert-base-chinese

<table>
<thead>
<tr><td>Model</td><td>Overall Span-Based F1 On Full Testset</td></tr>
</thead>
<tbody>
<tr><td>BERT-Linear</td><td>0.47296831955922863</td></tr>
<tr><td>BERT-Linear-CRF</td><td>0.537369759619329</td></tr>
<tr><td>BERT-Prompt</td><td>0.44231212097950406</td><tr>
</tbody>
</table>


<p id="5"> </p>

# Acknowledgement And Citations

<p id="5_1"> </p>

## People And Orgnizations

- BJTU-NLP

<p id="5_2"> </p>

## Third-Party Libraries

- pytorch
- transformers
- datasets
- seqeval
- ujson
- tqdm
- matplotlib
