"""
Utility functions for synthetic data.

Eg. Query Generation
"""
import re
import csv
import pandas as pd
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.infer_utils import generate_single_query

def write_to_csv_query_dset(csv_path:str, path_out:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, positive_and_negative=False) -> None:
    """Writes the (generated queries, document) pairs into a csv
    Inputs :
        - csv_path : the path of the csv dataset (must contain a column "text")
        - path_out : the path to save the CSV file
        - model : the pretrained T5 model for query generation
        - tokenizer : the model's tokenizer
        - device
        - positive_and_negative : True if the user wants to generative positive queries as well as hard negative queries for the retriever
    """

    df  =   pd.read_csv(csv_path)   # Load dataset

    # Init CSV file
    f       =   open(path_out, 'w')
    writer  =   csv.writer(f, delimiter="\t", lineterminator="\n")
    writer.writerow(["query"] + ["document"] + ["label"])

    for sample in tqdm(df['text']):
        doc             =   re.sub("\n", " ", sample)   # Format text
        pos_gen_queries =   generate_single_query(doc, model, tokenizer, device, positive=True)

        for q in pos_gen_queries:
            writer.writerow([q] + [doc] + [1])

        if positive_and_negative:
            neg_gen_queries =   generate_single_query(doc, model, tokenizer, device, positive=False)
            for q in neg_gen_queries:
                writer.writerow([q] + [doc] + [0])