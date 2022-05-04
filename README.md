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
<tr><td rowspan="5">Datasets</td></tr>
    <tr><td>CoNLL2003</td><td><a href="https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/tree/master/data/CoNLL2003_NER">
yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification</a></td></tr>
    <tr><td>OntoNotes5</td><td><a href="https://catalog.ldc.upenn.edu/LDC2013T19">LDC2013T19</a></td></tr>
    <tr><td>CCKS2019 Subtask 1</td><td><a href="https://tianchi.aliyun.com/dataset/dataDetail?dataId=92085">TIANCHI</a> (NER on Chinese medical documents.)</td></tr>
    <tr><td>NCBI-disease + s800</td><td><a href="https://github.com/dmis-lab/biobert">BioBERT</a> (NER on English medical doucments, Got these datasets from its download.sh)</td></tr>
<tr><td rowspan="3">Traning Tricks</td></tr>
    <tr><td>Gradient Accumulation</td><td></td></tr>
    <tr><td>Learning Rate Warmup</td><td></td></tr>
<tr><td rowspan="3">Misc</td></tr>
    <tr><td>Tokenizer from datasets</td><td>See myutils.py</td></tr>
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
    <td>Weight <br>Decay</td> 
    <td>Warmup <br>Ratio</td> 
    <td>Batch Size</td> 
    <td>Gradient <br>Accumulation</td> 
    <td>Clip Grad Norm</td>
    <td>Random Seed</td>
    </tr>
</thead>
<tbody>
<tr>
    <td>AdamW</td>
    <td>5e-3</td>
    <td>0.2</td>
    <td>1</td>
    <td>32</td>
    <td>1.0</td>
    <td>233</td>
</tr>
</tbody>
</table>

Training Epoches:
<table>
<thead>
<tr><td>Dataset</td><td>Full Data</td><td>Few Shot</td></tr>
</head>
<tbody>
<tr><td>CoNLL2003</td>
    <td rowspan="3">12</td> <td rowspan="4">30</td>
    </tr>
<tr><td>OntoNotes5(Chinese)</td></tr>
<tr><td>CCKS2019</td></tr>
<tr><td>NCBI-disease+s800</td><td>20</td></tr>
</body>
</table>

Learning Rates:

<table>
<thead>
<tr><td> </td>
    <td>CoNLL2003</td>
    <td>OntoNotes5</td>
    <td>CCKS2019</td>
    <td>NCBI-disease+s800</td>
    </tr>
</thead>
<tbody>
<tr><td>BiLSTM-Linear</td></td>
    <td colspan="2" rowspan="2">0.001</td>
    <td colspan="2" rowspan="2">NA</td>
    </tr>
<tr><td>BiLSTM-Linear-CRF</td></tr>
<tr><td>BERT-Linear</td>
    <td colspan="4" rowspan="2">0.0001</td></tr>
<tr><td>BERT-Linear-CRF</td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td rowspan="3">0.0001</td>
    <td>3e-5</td>
    <td colspan="2" rowspan="2">NA</td>
    </tr>
    <tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td>1e-5</td></tr>
    <tr><td>BERT(Prompt)</td>
    <td>3e-5</td>
    <td colspan="2" >0.0001</td></tr>

</tbody>
</table>

<p id="4_2"> </p>

## Model Parameters

<table>
<thead>
<tr><td>BERT Model</td> <td>Embedding Size(For models without BERT)</td> <td> LSTM Hidden Size </td> <td> LSTM Layers </td> </tr>
</thead>
<tbody>
<tr><td> bert-base-uncased(CoNLL2003,NCBI-disease+s800) </td> <td rowspan="2"> 256 </td> <td rowspan="2"> 256 </td> <td rowspan="2"> 2 </td></tr>
<tr><td>bert-base-chinese(OntoNotes5,CCKS2019)</td></tr>
</tbody>
</table>

<p id="4_3"> </p>

## Results

<p id="4_3_1"></p>

### Full Data Results

General datasets(ConLL2003, OntoNotes5(Chinese)).

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
    <td><b>0.9230769230769231</b></td>
    <td>99.70s</td></tr>

<tr><td rowspan="7">OntoNotes5<br>(Chinese)</td><td>BiLSTM-Linear</td>
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
    <td><b>0.7616389699353039</b></td>
    <td>800.23s</td></tr>
<tr><td>BERT(Prompt)</td>
    <td>0.7376454875023851</td>
    <td>485.56s</td></tr>

</tbody>
</thead>
</table>

Medical datasets, used general bert and medical bert.

<table>
<thead>
<tr><td>Dataset</td><td>BERT</td><td>Model</td><td>Overall Span-Based Micro F1</td></tr>
</thead>
<tbody>
<tr><td rowspan="6">CCKS2019<br>Subtask1</td>
    <td rowspan="3">bert-<br>base-<br>chinese</td>
    <td>BERT-Linear</td><td>0.8057400574005741</td></tr>
<tr><td>BERT-Linear-CRF</td><td><b>0.8119778310861113</b></td></tr>
<tr><td>BERT-Prompt</td><td>0.7684884784959654</td></tr>
<tr><td rowspan="3">medbert-<br>base-<br>chinese</td>
    <td>BERT-Linear</td><td>0.8201214508452324</td>
    </tr>
<tr><td>BERT-Linear-CRF</td><td><b>0.8221622063998691</b></td>
    </tr>
<tr><td>BERT-Prompt</td><td>0.7933091394485463</td></tr>

<tr><td rowspan="6">NCBI-disease<br>+s800</td>
    <td rowspan="3">bert-<br>base-<br>uncased</td>
    <td>BERT-Linear</td><td>0.7681962025316457</td></tr>
<tr><td>BERT-Linear-CRF</td><td><b></b></td></tr>
<tr><td>BERT-Prompt</td><td></td></tr>
<tr><td rowspan="3">
biobert-base-<br>cased-v1.2<br></td>
    <td>BERT-Linear</td><td></td>
    </tr>
<tr><td>BERT-Linear-CRF</td><td></td>
    </tr>
<tr><td>BERT-Prompt</td><td></td>
    </tr>

</tbody>
</table>

<p id="4_3_2"></p>

### Few Shot Results

Sampling <b>1%</b> data in trainset by fixed random seed. They used <b>the same hyper parameters</b> in full data experiments.

<b>Few Shot Test on CoNLL2003:</b>

Sampled 69 samples(total 6973). Here list the number of entities in few shot dataset:

```
{'MISC': 51, 'ORG': 51, 'PER': 59, 'LOC': 90}
```

<table>
<thead>
<tr><td rowspan="2">Model</td><td colspan="1">Overall Span-Based F1 On Full Testset</td></tr>
<tr><td>bert-base-uncased</td></tr>
</thead>
<tbody>
<tr><td>BERT-Linear</td><td>0.6778304852260387</td></tr>
<tr><td>BERT-Linear-CRF</td><td>0.6773130256876562</tr>
<tr><td>BERT-Prompt</td><td><b>0.7524185216492908</b></td></tr>
<tr><td>BERT-BiLSTM-Linear</td><td>0.037065541975802724</td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td><td>0.029508301201363885</td></tr>
</tbody>
</table>

<b>Few Shot Test on CCKS2019:</b>

Sampled 10 samples(total 1000). Here list the numbers of entities in few shot dataset:

```
{'手术': 9, '影像检查': 5, '疾病和诊断': 45, '解剖部位': 48, '实验室检验': 19, '药物': 10}
```

<table>
<thead>
<tr><td rowspan="2">Model</td><td colspan="2">Overall Span-Based F1 On Full Testset</td></tr>
<tr><td>bert-base-chinese</td><td>medbert-base-chinese</td></tr>
</thead>
<tbody>
<tr><td>BERT-Linear</td><td>0.43918064570513354</td><td>0.47296831955922863</td></tr>
<tr><td>BERT-Linear-CRF</td><td><b>0.47901807928346324</b></td><td><b>0.537369759619329</b></td></tr>
<tr><td>BERT-Prompt</td><td>0.0038852361028093247</td><td>0.43338090840399623</td><tr>
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
