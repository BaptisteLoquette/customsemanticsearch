"""
Inferences utility functions
"""

from typing import Dict, List
from torch import Tensor
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_batch_queries(batch:Dict[str, Tensor], model:T5ForConditionalGeneration, tokenizer:T5Tokenizer) -> List[Tensor]:
    """
    Generates queries from batch.

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
            generated_queries.append(gen_query.strip('query: '))

    return generated_queries

def generate_single_query(document:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, n_queries=3, positive=True) -> List[Tensor]:
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
    if positive:
        tokenized   =   tokenizer.encode_plus(f"generate_positive_query: {document}", max_length=512, padding=True, return_tensors="pt")
    else:
        tokenized   =   tokenizer.encode_plus(f"generate_negative_query: {document}", max_length=512, padding=True, return_tensors="pt")
    input_ids, attn_mask    =   tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)
    gen_sequences   =   model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        do_sample=True,
        max_length=96,
        top_p=0.95,
        num_return_sequences=n_queries
    )

    gen_queries =   tokenizer.batch_decode(gen_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return gen_queries