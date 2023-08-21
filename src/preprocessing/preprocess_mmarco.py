"""
Preprocess mmarco dataset for Query Generation
"""

from transformers import T5Tokenizer
from datasets import load_from_disk
import argparse
import os

from src.utils.preprocessing_utils import clean_and_format, tokenize_dset_query_gen

parser  =   argparse.ArgumentParser(
    prog="preprocess_mmarco",
    description="Preprocess unicamp-dl/mmarco dataset for Query Generation"
)
parser.add_argument('out', help="Output dir of the preprocessed files", type=str)
parser.add_argument('dset_path', help="Path of download mmarco dataset", type=str, default="../data/mmarco_queries_dset")
parser.add_argument('--tokenizer_path', help="Path to tokenizer", type=str, default="t5-small")
parser.add_argument('--n_training_samples', help="Number of training samples to train the model on. For everything type 'all'", type=int, default=500_000)
args    =   parser.parse_args()

if "__main__" == __name__:
    output_dir      =   args.out
    dset_path       =   args.dset_path
    tokenizer_path  =   args.tokenizer_path
    dset            =   load_from_disk(dset_path)
    n_samples       =   len(dset) if args.n_training_samples=='all' else int(args.n_training_samples)

    # Select the train & valid subsets
    dset    =   dset.select([i for i in range(n_samples)])
    dset    =   dset.map(clean_and_format)

    # Separate the dataset into a positive examples dataset and negative example datasets
    # This allows for training t5 to generate positive queries, and/or to generate hard negatives queries to ameliorate the retriever's accuracy
    negative_dset   =   dset.remove_columns("positive").rename_column("negative", "document")
    positive_dset   =   dset.remove_columns("negative").rename_column("positive", "document")

    # Tokenize the datasets
    tokenizer       =   T5Tokenizer.from_pretrained(tokenizer_path)
    positive_dset   =   positive_dset.map(tokenize_dset_query_gen, fn_kwargs={"tokenizer": tokenizer})
    positive_dset   =   positive_dset.remove_columns(["query", "document"])
    negative_dset   =   negative_dset.map(tokenize_dset_query_gen, fn_kwargs={"tokenizer": tokenizer})
    negative_dset   =   negative_dset.remove_columns(["query", "document"])   # Remove str columns

    # Shuffle datasets
    positive_dset   =   positive_dset.shuffle()
    negative_dset   =   negative_dset.shuffle()

    # Save Datasets
    positive_dset.save_to_disk(os.path.join(output_dir, "positive"))
    negative_dset.save_to_disk(os.path.join(output_dir, "negative"))