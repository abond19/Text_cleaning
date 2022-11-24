#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date       : 2022-01-08 22:39:29
# @Author     : Chenghao Mou (mouchenghao@gmail.com)
# @Description: Self-deduplication with `datasets`

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
from deduplicate import INTERNAL_HASH
from deduplicate.util import hashing
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)


def main(conf: str) -> None:

    with open(conf, "r") as f:
        conf = yaml.safe_load(f.read())

    if conf["load_from_disk"]["path"]:
        fs: AbstractFileSystem = None
        if conf["load_from_disk"]["gcs"]:
            fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
        ds = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
    else:
        ds = load_dataset(**conf["load_dataset"])

    logger.info(f"Done loading {len(ds)} records")

    if not os.path.exists(conf["cache"]):
        os.makedirs(conf["cache"], exist_ok=True)

    if not os.path.exists(conf["output"]):
        os.makedirs(conf["output"], exist_ok=True)
    """
    urls = ds.map(
        lambda x: {
            "url": x["meta"]["headers"]["warc-target-uri"],
            "text": x["text"].replace("\n", " "),
        },
        num_proc=conf["num_proc"],
        desc="Extracting URLs",
    )
    """
    urls = ds.map(
        lambda x: {
            "text": x["text"].replace("\n", " "),
        },
        num_proc=conf["num_proc"],
        desc="Extracting URLs",
    )
    
    #logger.info(
    #   f"Extracted URLs: {len(urls['url'])}, Unique URLs: {len(set(urls['url']))}"
    #)
    
    #urls = DatasetDict({"train": urls})
    
    urls.rename_column(original_column_name="Unnamed: 0", new_column_name="id")
    
    print(urls)

    # Save text data for substring deduplication
    
    urls['train'].to_csv(
        os.path.join(conf["output"], "text.csv"),
        num_proc=conf["num_proc"],
        index=False,
        header=False,
        columns=["text"],
    )
    urls['train'].to_csv(
        os.path.join(conf["output"], "ids.csv"),
        num_proc=conf["num_proc"],
        index=False,
        header=False,
        columns=["Unnamed: 0"],
    )
    
    del urls

    logger.info(f"Start hashing {len(ds)} records")
    if conf["ignore_punctuation"]:
        assert (
            conf["tokenization"] != "punctuation"
        ), f"Cannot ignore punctuation when tokenization is set to `punctuation`"

    ds = ds.map(
        hashing,
        fn_kwargs={
            "tokenization": conf["tokenization"],
            "window_size": conf["window_size"],
            "column": conf["text_column"],
            "ignore_punctuation": conf["ignore_punctuation"],
            "lowercase": conf["lowercase"],
            "output": INTERNAL_HASH,
        },
        num_proc=conf["num_proc"],
        desc="Hashing",
    )
    #print(ds['train']['__dedup_hash__'])
    logger.info(f"Done hashing {len(ds)} records")
    HASH = '__dedup_hash__'
    logger.info(f"Start querying {len(ds)} records")
    ds = ds['train']
    matches = simhash.find_all(
        tqdm(ds['__dedup_hash__'], total=len(ds)),
        conf["num_blocks"],
        conf["hamming_distance"],
    )
    logger.info(f"Done querying {len(ds)} records, found {len(matches)} matches")
    graph = defaultdict(dict)
    dist = Counter()
    examples = defaultdict(set)
    for x, y in matches:
        graph[x][y] = True
        graph[y][x] = True
        dist[simhash.num_differing_bits(x, y)] += 1
        if len(examples[simhash.num_differing_bits(x, y)]) < 3:
            examples[simhash.num_differing_bits(x, y)].add((x, y))

    logger.info(f"Hash difference distribution: {dist}")

    hash2ids: Dict[int, Set[str]] = defaultdict(set)
    hashes: Set[int] = set(ds['__dedup_hash__'])
    hash2cluster: Dict[int, int] = {}
    visited: Set[int] = set()
    cluster_id: int = 0
    
    for element in ds:
        hash = element['__dedup_hash__']
        hash2ids[hash].add(element['Unnamed: 0'])
        

    seen = set()
    with open(os.path.join(conf["output"], "matches.tsv"), "w") as o:
        o.write(f"id1\tid2\tdiff\n")
        for x, y in matches:
            for id1, id2 in product(hash2ids[x], hash2ids[y]):
                if id1 == id2:
                    continue
                if tuple(sorted((id1, id2))) in seen:
                    continue
                seen.add(tuple(sorted((id1, id2))))
                o.write(f"{id1}\t{id2}\t{simhash.num_differing_bits(x, y)}\n")

    # print some match samples

    datasets.disable_progress_bar()
    example_text = []
    for diff in tqdm(examples):
        for x, y in examples[diff]:
            records = []
            ids = hash2ids[x]
            ids.update(hash2ids[y])
            for text in ds.filter(
                lambda x: x["Unnamed: 0"] in ids,
                num_proc=conf["num_proc"],
            )["text"]:
                records.append(text)
            example_text.append((diff, records))

    datasets.enable_progress_bar()
    with open(os.path.join(conf["output"], "examples.txt"), "w") as o:
        for diff, records in example_text:
            o.write("*" * 80 + "\n")
            for text in records:
                o.write(f"\n({diff}) {text}\n")

    while hashes:
        hash = hashes.pop()
        if hash in visited:
            continue

        # BFS to find the cluster
        if hash not in graph:
            hash2cluster[hash] = -1
            continue

        q = deque([hash])
        visited.add(hash)
        hash2cluster[hash] = cluster_id

        while q:
            node = q.popleft()
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                q.append(neighbor)
                hash2cluster[neighbor] = cluster_id

        cluster_id += 1

    logger.info(f"Found {cluster_id} clusters and {len(graph)} hashes")

    with open(os.path.join(conf["output"], "clusters.tsv"), "w") as o:
        o.write(f"id\thash\tcluster\n")
        for element in ds:
            hash = element['__dedup_hash__']
            o.write(f"{element['Unnamed: 0']}\t{hash}\t{hash2cluster.get(hash, -1)}\n")

    logger.info("Done!")


if __name__ == "__main__":

    typer.run(main)
