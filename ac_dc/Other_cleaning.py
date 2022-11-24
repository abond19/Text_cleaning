import os
import re
from tqdm import tqdm
import warnings

from datasets import load_dataset, load_from_disk
from datasets import Dataset
from itertools import product
import logging
from collections import Counter, defaultdict, deque
from typing import Dict, Set
from multiprocessing import Pool, cpu_count

import gcsfs
import simhash
import typer
import yaml
import datasets
from datasets import load_dataset, DatasetDict
from datasets.load import load_from_disk
from fsspec.spec import AbstractFileSystem

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

def check_num_proc(num_proc: int = -1) -> int:
    """
    Check the number of processors. Return a safe-checked value.

    Parameters
    ----------
    num_proc : int, optional
        Number of processors to use, by default -1

    Returns
    -------
    int
        Number of processors to use

    Raises
    ------
    ValueError
        If the input exceeds the number of processors available
    """
    maximum: int = cpu_count()
    if num_proc > maximum:
        raise ValueError(
            f"{num_proc} exceeds the maximum number ({maximum}) of processors"
        )

    if num_proc == -1:
        num_proc = maximum
    else:
        print(f"Using {num_proc} processors out of {maximum} can be slow")

    return num_proc

def remove_urls(text):
    line = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',"",text)
    return line

def remove_lists(line):
    pattern = "\|[ ]*[\-\-]*[\w ışöüğÜ'\-çİÖ©]+ \|"
    match = re.findall(pattern, line, re.DOTALL)
    if match is not []:
        return re.sub(pattern, "", line, re.DOTALL)
    else:
        return line

def is_allowed(c):
    if c.isalpha():
        return True
    if c == " " or c == "\t" or c == "\n":
        return True
    if c == "." or c == "," or "'" or "\"" or "!" or "?":
        return True
    if c in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        return True
    return False

def remove_non_letters(line):
    return "".join(c for c in line if is_allowed(c))
    

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def to_remove(sentence):
    if len(sentence) <= 50:
        return True
    # Handle case with bad tokenization
    num_spaces = sentence.count(" ")
    num_letters = len(sentence) - num_spaces
    if num_spaces != 0 and num_letters / num_spaces < 3:
        return True
    #Handle case with weird encoding
    weird_symbols = ['ý', 'ð', 'þ', 'Ý', 'Þ']
    if any(symbol in sentence for symbol in weird_symbols):
        return True
    return False

def clean_sentence(sentence):
    # Remove Chinese characters
    if to_remove(sentence):
        return ""
    
    result = remove_lists(sentence)
    
    result = remove_non_letters(result)
    
    result = remove_urls(result)
    
    result = re.sub(u'[\u4E00-\u9FA5]', "", result)
    
    # Remove spaces before commas
    result = re.sub(" ,", ",", result)
    
    # Remove duplicate spaces
    result = re.sub(" {2,}", " ", result)
    
    # Put spaces for camel case
    result = " ".join(camel_case_split(result)).strip()
    result = result.replace('\n', ' ').replace('\r', '')
    #sentence['text'] = result
    return result

if __name__ == "__main__":
    output_file = open("ProperMC4/finished.txt", "w")
    
    
   # df = pd.read_csv("deduplicate/outputs/tr/text.csv")
    
    """
    conf = "./deduplicate/conf/self_deduplicate_tr.yaml"

    with open(conf, "r") as f:
        conf = yaml.safe_load(f.read())
        print("Loaded configuration")

    if conf["load_from_disk"]["path"]:
        fs: AbstractFileSystem = None
        if conf["load_from_disk"]["gcs"]:
            fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
        ds = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
        print("Finished loading dataset")
    else:
        ds = load_dataset(**conf["load_dataset"])
    """
    #ds = ds['text']
    #print(ds)
    
    pool = Pool(check_num_proc())
    
    print("Started cleaning")
    with open("ProperMC4/mc4.txt") as f:
        for line in tqdm(pool.imap_unordered(clean_sentence, f, chunksize=100), total=62775992):
        #for line in tqdm(f, total=1344075912):
            #cleaned = clean_sentence(line)
            if line != "":
                output_file.write(line + "\n")
    
    #for line in tqdm(ds['text'], total=11861338):
    #    cleaned = clean_sentence(line)
    #    if line != "":
    #        output_file.write(line + "\n")

    #with open("samples/sketchengine_sample.txt", "r") as f:
    #    for line in f:
    #        cleaned = clean_sentence(line)
    #        if cleaned != "":
    #            output_file.write(cleaned + "\n")
    
    #for line in tqdm(pool.imap_unordered(clean_sentence, ds['text'], chunksize=100), total=11861338):
    #    if line != "":
    #        output_file.write(line + "\n")
    
    #ds.map(clean_sentence, batched=True, num_proc=check_num_proc())
    
    #ds.set_format('pandas')
    
    #ds.to_csv("outputs/processed.csv")
    
    #new_df = pd.DataFrame(columns=["text"])
    
    #for index, line in tqdm(df.iterrows()):
    #    new_line = clean_sentence(line.iloc[0])
    #    to_append = {"text" : new_line}
    #    new_df = new_df.append(to_append, ignore_index=True)
    
    
    #for element in new_df['text']:
    #    if len(element) > 50:
    #        output_file.write(element + "\n")
    
    output_file.close()
    