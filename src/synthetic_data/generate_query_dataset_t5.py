"""
Generates a Pseudo-Labeled dataset of (queries, document) pairs from cc_news dataset.
"""
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.synthetic_data_utils import write_to_csv_query_dset_t5
import torch

parser  =   argparse.ArgumentParser(
    prog="generate_query_dataset_csv",
    description="Generate a queries dataset on any csv dataset"
)
parser.add_argument('csv_path', help="Input path of the CSV dataset, must contain a columns 'text' where ", type=str)
parser.add_argument('path_out', help="Output path to save the CSV generated dataset", type=str)
parser.add_argument('pretrained_model_path', help="Path to pretrained T5's states to initialize the model with", type=str)
parser.add_argument('tokenizer_path', help="Path to model's tokenizer", type=str)
parser.add_argument('--n_queries', help="Number of queries to generate", type=int)
parser.add_argument('--pos_and_neg', help="True if the user wants to generative positive queries as well as hard negative queries for the retriever", type=bool, default=False)
parser.add_argument('--extractive_sum', help="True if the user wants to perform extractive summarization on documents longer than max length (increases time/iteration)", type=bool, default=False)

if "__main__" == __name__:
    args                    =   parser.parse_args()
    csv_path                =   args.csv_path
    path_out                =   args.path_out
    pretrained_model_path   =   args.pretrained_model_path
    tokenizer_path          =   args.tokenizer_path
    positive_and_negatives  =   args.pos_and_neg
    n_queries               =   args.n_queries
    extractive_sum          =   args.extractive_sum

    # Init model & tokenizer, move model to GPU or CPU
    model       =   T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    tokenizer   =   T5Tokenizer.from_pretrained(tokenizer_path)
    device      =   torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model       =   model.to(device)

    write_to_csv_query_dset_t5(csv_path, path_out, model, tokenizer, device, positive_and_negative=positive_and_negatives, n_queries=n_queries, extractive_sum=extractive_sum)    # Generate and write to csv