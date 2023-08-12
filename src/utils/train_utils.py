"""
Training Utility Functions
"""

from typing import List, Dict
import re
import pandas as pd
from tqdm import tqdm
from torch import Tensor
from evaluate import load
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.utils.infer_utils import generate_single_query

rouge_metric    =   load('rouge')

def compute_rouge_score_train(gt_target_ids:Tensor, gen_queries:List[str], tokenizer:T5Tokenizer) -> Dict[str, float]:
    """Compute rouge score during training (from input ids as the dataset is formatted)

    Inputs :
        - gt_target_ids : the ground_truth input_ids of the queries
        - gen_queries : the generated queries
        - tokenizer : the model's tokenizer

    Ouputs :
        rouge_score : the ROUGE's scores
    """
    gt_queries  =   tokenizer.batch_decode(gt_target_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)

    rouge_score =   rouge_metric.compute(predictions=gen_queries,
                                         references=gt_queries)
    return rouge_score

def compute_rouge_score(test_path:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, positive=True) -> Dict[str, float]:
    """Compute rouge score from unformated mmarco dataset

    Inputs :
        - test_path : path to the test set of mmarco
        - model : path to the model's pretrained weights
        - tokenizer : the model's tokenizer
        - device
        - positive : calculates on positive or negatives ?
        
    Ouputs :
        rouge_score : the ROUGE's scores
    """
    dset    =   pd.read_csv(test_path)
    gt_q    =   []
    gen_q   =   []
    for doc_pos, doc_neg, gt_query in tqdm(zip(dset['positive'], dset['query'])):
        if positive:
            gen_q.extend(generate_single_query(re.sub("\n", " ", doc_pos), model, tokenizer, positive=positive, device=device, greedy=True))
        else:
            gen_q.extend(generate_single_query(re.sub("\n", " ", doc_neg), model, tokenizer, positive=positive, device=device, greedy=True))
        gt_q.append(gt_query)
    rouge_score =   rouge_metric.compute(predictions=gen_q,references=gt_q)
        
    return rouge_score