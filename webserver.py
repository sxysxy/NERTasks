from lib2to3.pgen2.tokenize import tokenize
from wsgiref.simple_server import WSGIServer
import flask
from flask.globals import request
import sys
import argparse
import ujson as json
from myutils import get_saved_model
from transformers import BertTokenizer
from ner_models import INERModel
from myutils import get_datasets, map_tagname_bio2io
import random
from myutils import NERDatasetsConfigs
import copy
import pdb
import re

def do_ner(tk : BertTokenizer, model : INERModel, text : str, tag_names, language = 'en'):
    if language == 'en':
        text = text.split(' ')
    elif language == 'ch':
        text = list(text.replace(' ',''))
    else:
        raise RuntimeError("Unknown language")
   # pdb.set_trace()
    lent = len(text)
    sp_start = 0
    result = []
    wids = []
    while sp_start < lent:
        sub_text = text[sp_start:sp_start+510]
        #print(sub_text)
        tokenized = tk.batch_encode_plus([sub_text], add_special_tokens=True, is_split_into_words=True, return_attention_mask=False, return_tensors='pt', return_token_type_ids=False)
        sub_result = model.decode(**{
            "input_ids" : tokenized["input_ids"]
        })[0]
        wid = [wi + sp_start for wi in filter(lambda x : x != None ,tokenized.word_ids()) ]
        wids.extend(wid)
        result.extend(sub_result)
        sp_start += len(sub_text)
    tagged = [ tag_names[tid] for tid in result ]
    real_tagged = []
    last_wid = -1
    
    xi = 0
    for wi in wids:
        if wi != None and wi != last_wid:
            real_tagged.append(tagged[xi])
            last_wid = wi
        xi += 1
    #print(len(real_tagged))
    #print(len(text))    
    assert len(real_tagged) == len(text)
    #print(list(zip(text, real_tagged)))
    result = list(zip(text, real_tagged))
    ent_result = []
    i = 0
    ls = len(result)
   # pdb.set_trace()
    while i < ls:
        if result[i][1] != 'O':
            j = i+1
            tag = result[i][1][2:]
            if result[i][1].startswith("B"):
                while j < ls and result[j][1].startswith("B") and result[j][1][2:] == tag:
                    j += 1
                while j < ls and result[j][1].startswith("I") and result[j][1][2:] == tag:
                    j += 1
            elif result[i][1].startswith("I"):
                while j < ls and result[j][1].startswith("I") and result[j][1][2:] == tag:
                    j += 1
            ents_tokens = [e[0] for e in result[i:j]]
            if language == 'en':
                ent = ' '.join(ents_tokens)
            else:
                ent = ''.join(ents_tokens)
            ent_result.append((ent, tag))
            i = j
        else:
            j = i + 1 
            while j < ls and result[j][1] == 'O':
                j += 1
            o_tokens = [o[0] for o in result[i:j]]
            if language == 'en':
                ot = ' '.join(o_tokens)
            else:
                ot = ''.join(o_tokens)
            ent_result.append((ot, "O"))
            i = j
    return ent_result

datasets_cached = {}
def get_dataset_cached(dataset_name):
    global datasets_cached
    if dataset_name in datasets_cached:
        return datasets_cached[dataset_name]
    datasets_cached[dataset_name] = get_datasets(f"{dataset_name}-base")
    return datasets_cached[dataset_name]

def is_english(dsname):
    return dsname == "ncbi-disease"

app = flask.Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template("index.html")

@app.route('/NER', methods=['POST'])
def ner():
    data = json.loads(request.get_data())
    tk, model = get_saved_model(data["dataset_name"], data["model_name"])
    tag_names = NERDatasetsConfigs.configs[data["dataset_name"]]["tag_names"]
    if data["model_name"] == "BERT-Prompt":
        #tag_names = map_tagname_bio2io(tag_names)
        tag_names = [""] * len(model.tag2idx)
        for k in model.tag2idx:
            tag_names[model.tag2idx[k]] = k
    ner_result = do_ner(tk, model, data["text"], tag_names, 'en' if is_english(data["dataset_name"]) else 'ch')
    return json.dumps(ner_result)
    #eturn flask.render_template('tagged_result.html', result=ner_result)

@app.route('/sample', methods=['POST'])
def sample():
    data = json.loads(request.get_data())
    ds = get_dataset_cached(data["dataset_name"])
    sample = random.choice(ds["train"])["tokens"]
    if is_english(data["dataset_name"]):
        sample = ' '.join(sample)
    else:
        sample = ''.join(sample)
    return json.dumps( {
        "sample" :  sample
    } )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action='store_true', default=False)
    parser.add_argument("--port", type=int, default=23333)
    args = parser.parse_args()

    get_dataset_cached("ccks2019")
    get_dataset_cached("ncbi-disease")

    if not args.release:
        app.run(host="0.0.0.0", port=args.port, debug=True)
    else:
        WSGIServer(('', args.port), app).serve_forever()