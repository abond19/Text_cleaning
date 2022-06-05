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

dataset = load_dataset('mc4', "tr", split="train", cache_dir="./mc4")

dataset.save_to_disk(dataset_path='./mc4_downloaded')

print(dataset.shape)

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
              
ds.set_format("pandas")
#df = ds['train'][:]
print(ds.shape)
ds.to_csv('./mc4_downloaded/train_mc4.csv')