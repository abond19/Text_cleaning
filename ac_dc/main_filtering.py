"""Filtering."""

from multiprocessing import cpu_count

import argparse

from datasets import load_dataset

from filtering import DatasetFiltering

import gcsfs
import simhash
import typer
import yaml
import datasets
from datasets import load_dataset, DatasetDict
from datasets.load import load_from_disk
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm


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


def parseArgs():
    parser = argparse.ArgumentParser(description="Filtering.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mc4",
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="tr",
        help="Name of the dataset config to pass.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="'load_dataset' returns all files that match the Unix style pattern passed by 'data_files'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split of the dataset to consider.",
    )
    parser.add_argument(
        "--lang_dataset_id",
        type=str,
        default="tr",
        help="ID of the language in which the dataset is written.",
    )
    parser.add_argument(
        "--path_fasttext_model",
        type=str,
        default="lid.176.bin",
        help="Path to the Fasttext model used for language identification.",
    )
    parser.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="spiece.tokenizer.tr.json",
        help="Path to the Sentence Piece model used to tokenize text for perplexity scores.",
    )
    parser.add_argument(
        "--path_kenlm_model",
        type=str,
        default="3-gram.trwiki.arpa.bin",
        help="Path to the KenLM model used to compute perplexity scores.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processes for multiprocessing. Default at the number of processors available.",
    )
    parser.add_argument(
        "--path_dir_save_dataset",
        type=str,
        default="../dataset_filtered/",
        help="Path to the directory where the filtered version of the dataset will be saved.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()

    #dataset = load_dataset(
    #    args.dataset_name,
    #    args.config_name,
    #    data_files=args.data_files,
    #    split=args.split,
    #    use_auth_token=True, 
    #    cache_dir="./mc4"
    #)
    """
    conf = "./deduplicate/conf/self_deduplicate_tr.yaml"

    with open(conf, "r") as f:
        conf = yaml.safe_load(f.read())

    if conf["load_from_disk"]["path"]:
        fs: AbstractFileSystem = None
        if conf["load_from_disk"]["gcs"]:
            fs = gcsfs.GCSFileSystem(project=conf["load_from_disk"]["gcs"])
        dataset = load_from_disk(conf["load_from_disk"]["path"], fs=fs)
    else:
        dataset = load_dataset(**conf["load_dataset"])
    """
    dataset = load_dataset("text", data_files="./ProperMC4/mc4.txt", cache_dir="./MC4")
    
    print("Finished loading dataset")
    dataset_filtering = DatasetFiltering(
       dataset=dataset,
        lang_dataset_id=args.lang_dataset_id,
        path_fasttext_model=args.path_fasttext_model,
        path_sentencepiece_model=args.path_sentencepiece_model,
        path_kenlm_model=args.path_kenlm_model,
        num_proc=check_num_proc(args.num_proc),
        path_dir_save_dataset=args.path_dir_save_dataset,
    )
    print("Created DatasetFiltering")
    dataset_filtering.modifying_documents()
    dataset_filtering.save_dataset()
    print("Finished modifying documents")
    dataset_filtering.filtering()
    print("Finished filtering")
    dataset_filtering.save_dataset()


if __name__ == "__main__":
    main()
