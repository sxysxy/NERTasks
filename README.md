# NERTasks

<h3><b> A simple NER framework. </b> </h3>

It implements:

<table>
<tbody>
<tr><td rowspan="5">Models</td></tr>
    <tr><td>BiLSTM-Linear</td></tr>
    <tr><td>BiLSTM-Linear-CRF</td></tr>
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

# Requirements:

Linux(Tested)/Windows(Not Tested) with Nvidia GPUs, or MacOS(no GPU acceleration). 

# Install dependencies.

Recommend to use conda create a python environment, python==3.9. For example:

```
conda create -n NER python=3.9
```

And run the bash script. If you are using windows, change its extname to .bat.

```
./install_dependencies.sh
```

# How to prepare datasets

For some reason(copyright and some other things), I can't directly provide datasets to you. You should get the access to these datasets by yourself and put them in specified format into 'assert/raw_datasets' folder, see [here](assets/README.md).