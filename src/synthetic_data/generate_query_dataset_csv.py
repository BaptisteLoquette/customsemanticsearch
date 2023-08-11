"""
Generates a Pseudo-Labeled dataset of (queries, document) pairs from cc_news dataset.
"""
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.synthetic_data_utils import write_to_csv_query_dset
import torch

parser  =   argparse.ArgumentParser(
    prog="generate_query_dataset_cc_news",
    description="Generate a queries dataset on cc_news dataset"
)
parser.add_argument('csv_path', help="Input path of the CSV dataset, must contain a columns 'text' where ", type=str)
parser.add_argument('path_out', help="Output path to save the CSV generated dataset", type=str)
parser.add_argument('pretrained_model_path', help="Path to pretrained T5's states to initialize the model with", type=str)
parser.add_argument('tokenizer_path', help="Path to model's tokenizer", type=str)
parser.add_argument('--pos_and_neg', help="True if the user wants to generative positive queries as well as hard negative queries for the retriever", type=bool, default=False)
                
if "__main__" == __name__:
    args                    =   parser.parse_args()
    csv_path                =   args.csv_path
    path_out                =   args.path_out
    pretrained_model_path   =   args.pretrained_model_path
    tokenizer_path          =   args.tokenizer_path
    positive_and_negatives  =   args.pos_and_neg

    # Init model & tokenizer, move model to GPU or CPU
    model       =   T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
    tokenizer   =   T5Tokenizer.from_pretrained(tokenizer_path)
    device      =   torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model       =   model.to(device)

    write_to_csv_query_dset(csv_path, path_out, model, tokenizer, device, positive_and_negative=positive_and_negatives)    # Generate and write to csv