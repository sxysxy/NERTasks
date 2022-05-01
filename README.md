# NERTasks

# Contents
## [Waht's It?](#0)
## [1. Requrements](#1)
## [2. Install Dependencies](#2)
## [3. How To Prepare Datasets](#3)
## [4. Experiments](#4)
### [&nbsp; 4.1 Hyper Parameters](#4_1)
### [&nbsp; 4.2 Model Parameters](#4_2)
### [&nbsp; 4.3 Results](#4_3)

<p id="0"> </p>

# What's It? 

<h4><b> A simple NER framework. </b> </h4>

It implements:

<table>
<tbody>
<tr><td rowspan="7">Models</td></tr>
    <tr><td>BiLSTM-Linear</td></tr>
    <tr><td>BiLSTM-Linear-CRF</td></tr>
    <tr><td>BERT-Linear</td></tr>
    <tr><td>BERT-BiLSTM-Linear</td></tr>
    <tr><td>BERT-BiLSTM-Linear-CRF</td></tr>
    <tr><td>BERT(Prompt)</td></tr>
<tr><td rowspan="4">Datasets</td></tr>
    <tr><td>CoNLL2003</td></tr>
    <tr><td>OntoNotes5</td></tr>
    <tr><td>CMeEE</td></tr>
<tr><td rowspan="4">Traning Trick</td></tr>
    <tr><td>Gradient Accumulation</td></tr>
    <tr><td>Learning Rate Warmup</td></tr>
    <tr><td>Label Smooth</td></tr>
<tr><td rowspan="3">Misc</td></tr>
    <tr><td>Tokenizer from datasets</td></tr>
    <tr><td>NER Metrics</td></tr>
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
<tr><td> </td>
    <td>Optimizer</td>
    <td>Learning Rate</td> 
    <td>Weight Decay</td> 
    <td>Warmup Ratio</td> 
    <td>Label Smoothing</td> 
    <td>Batch Size</td> 
    <td>Gradient Accumulation</td> 
    <td>Total Epoches</td>
    <td>Random Seed</td>
    </tr>
</thead>
<tbody>
<tr><td>BiLSTM-Linear</td>
        <td rowspan="6">Adam</td>
        <td rowspan="2"> 0.001 </td>
        <td rowspan="6"> 5e-3 </td>
        <td rowspan="3"> 0.2 </td>
        <td rowspan="6"> None </td>
        <td rowspan="6"> 1 </td>
        <td rowspan="6"> 32 </td>
        <td rowspan="6"> 12 </td>
        <td rowspan="6"> 233 </td></tr>
    <tr><td>BiLSTM-Linear-CRF</td></tr>
    <tr><td>BERT-Linear</td>
        <td rowspan="5">0.0001</td>
        </tr>
    <tr><td>BERT-BiLSTM-Linear</td>
        <td rowspan="4">None</td></tr>
    <tr><td>BERT-BiLSTM-Linear-CRF</td></tr>
    <tr><td>BERT(Prompt)</td></tr>
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

<table>
<thead>
<tr>
    <td> Dataset </td>
    <td> Model </td>
    <td> Overall Span-Based F1 </td>
</tr>
</thead>
<tbody>
<tr><td rowspan="6">CoNLL2003</td><td>BiLSTM-Linear</td>
    <td>0.6617193523515805</td></tr>
<tr><td>BiLSTM-Linear-CRF</td>
    <td>0.6955084580983861</td></tr>
<tr><td>BERT-Linear</td>
    <td>0.8985507246376812</td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td></td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td>0.8751660026560425</td></tr>
<tr><td>BERT(Prompt)</td>
    <td></td></tr>

<tr><td rowspan="6">OntoNotes5(Chinese)</td><td>BiLSTM-Linear</td>
    <td></td></tr>
<tr><td>BiLSTM-Linear-CRF</td>
    <td></td></tr>
<tr><td>BERT-Linear</td>
    <td></td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td></td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td></td></tr>
<tr><td>BERT(Prompt)</td>
    <td></td></tr>
</tbody>
</thead>
</table>

CMeEE is evaluated by official judger: [https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414)

<table>
<thread>
<tr><td>Dataset</td><td>Model</td><td> F1 </td></tr>
</thread>
<tbody>
<tr><td rowspan="6">CMeEE</td><td>BiLSTM-Linear</td>
    <td></td></tr>
<tr><td>BiLSTM-Linear-CRF</td>
    <td></td></tr>
<tr><td>BERT-Linear</td>
    <td></td></tr>
<tr><td>BERT-BiLSTM-Linear</td>
    <td></td></tr>
<tr><td>BERT-BiLSTM-Linear-CRF</td>
    <td></td></tr>
<tr><td>BERT(Prompt)</td>
    <td></td></tr>
</tbody>
</table>
