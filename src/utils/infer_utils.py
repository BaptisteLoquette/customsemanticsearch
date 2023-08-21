"""
Inferences utility functions
"""

from typing import Dict, List
from torch import Tensor
import torch
import numpy as np
from nltk import sent_tokenize
from summarizer import Summarizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

summarize_model =   Summarizer()

def generate_batch_queries(batch:Dict[str, Tensor], model:T5ForConditionalGeneration, tokenizer:T5Tokenizer) -> List[Tensor]:
    """
    Generates queries from batch of source_ids and attention_mask

    Inputs :
        - batch : tokenized batch of input documents
        - model : Pretrained T5 model
        - tokenizer : model's tokenizer

    Outputs :
        - generated_queries : list of generated queries
    """
    generated_queries   =   []
    model.eval()
    with torch.no_grad():
        for idx in range(len(batch)-1):
            out =   model.generate(
                input_ids=batch["source_ids"][idx],
                attention_mask=batch["source_attention_mask"][idx],
                max_length=96,
                early_stopping=True,
            ).squeeze()

            gen_query   =   tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_queries.append(gen_query)

    return generated_queries

def generate_single_query(document:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, n_queries=3, positive=True, greedy=False, extractive_sum=False, max_length=512) -> List[Tensor]:
    """Generate n_queries queries from a single document
    Inputs :
        - document : the document's text
        - model : the pretrained T5 model for query generation
        - tokenizer : the tokenizer's model
        - device
        - positive : Bool to specify if we want to generate a "positive" query (will be considered as ground truth for the training of the Retriever) or a hard negative
    Outputs :
        - gen_queries : the list of generated queries
    """
    model.eval()
    if positive:
        document    =   f"generate_positive_query: {document.encode('ascii', 'ignore').decode('ascii')}"
    else:
        document    =   f"generate_negative_query: {document.encode('ascii', 'ignore').decode('ascii')}"

    sent_length =   [len(tokenizer.tokenize(sent)) for sent in sent_tokenize(document)] # Computes sentences length, needed to find an optimal number of sentences if summarization

    if sum(sent_length) > max_length and extractive_sum:
        # Average an optimal number of sentences
        average_sent_len    =   np.mean(sent_length)
        ideal_number_sents  =   np.ceil(max_length/average_sent_len)
        document            =   summarize_model(document, num_sentences=int(ideal_number_sents))

    tokenized   =   tokenizer.encode_plus(document, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    input_ids, attn_mask    =   tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)

    if greedy:
        gen_sequences   =   model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_length=96
        )
    else:
        gen_sequences   =   model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            do_sample=True,
            max_length=96,
            top_p=0.95, # nucleus sampling to encourage variety in the generated queries
            num_return_sequences=n_queries
        )

    gen_queries =   tokenizer.batch_decode(gen_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return gen_queries