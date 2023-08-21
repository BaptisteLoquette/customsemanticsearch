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

def compute_rouge_score_from_csv(test_path:str, model:T5ForConditionalGeneration, tokenizer:T5Tokenizer, device, positive=True) -> Dict[str, float]:
    """Compute rouge score from unformated mmarco dataset

    Inputs :
        - test_path : path to the test set of mmarco (only positive or only negative)
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
    for doc_pos, doc_neg, gt_query in tqdm(zip(dset['positive'], dset['negative'], dset['query'])):
        if positive:
            gen_q.extend(generate_single_query(re.sub("\n", " ", doc_pos), model, tokenizer, positive=positive, device=device, greedy=True))
        else:
            gen_q.extend(generate_single_query(re.sub("\n", " ", doc_neg), model, tokenizer, positive=positive, device=device, greedy=True))
        gt_q.append(gt_query)
    rouge_score =   rouge_metric.compute(predictions=gen_q,references=gt_q)
        
    return rouge_score