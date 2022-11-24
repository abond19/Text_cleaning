from datasets import load_dataset, load_from_disk
from datasets import Dataset
from itertools import product
import logging
import os
from collections import Counter, defaultdict, deque
from typing import Dict, Set

import gcsfs
import simhash
import typer
import yaml
import datasets
from datasets import load_dataset, DatasetDict
from datasets.load import load_from_disk
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm

#dataset = load_dataset('mc4', "tr", split="train", cache_dir="./mc4_2")

#dataset.save_to_disk(dataset_path='./mc4_downloaded')

#print(dataset.shape)

def clean_sentence(sentence):
    sentence['text'] =  sentence['text'].replace("\n", " ").replace("\r", "").replace(",", "")
    return sentence

conf = "./deduplicate/conf/self_deduplicate_tr.yaml"

with open(conf, "r") as f:
    conf = yaml.safe_load(f.read())

if conf["load_from_disk"]["path"]:
    fs: AbstractFileSystem = None
    if conf["load_from_disk"]["gcs"]:
        fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
    ds = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
else:
    ds = load_dataset(**conf["load_dataset"])
              
#df = ds['train'][:]
print(ds.shape)

#for index, line in tqdm(ds.iterrows()):
#    new_line = clean_sentence(line.iloc[0])
#    #print(new_line)
#    ds.iloc[index]['text'] = new_line
    
ds.map(clean_sentence)
ds.set_format("pandas")

#ds = ds['text']

ds.to_csv('./mc4_downloaded/train_mc4_2.csv')