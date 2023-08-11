"""
Training Utility Functions
"""

from typing import List, Dict
from torch import Tensor
from evaluate import load
from transformers import T5Tokenizer

rouge_metric    =   load('rouge')

def compute_rouge_score(gt_target_ids:Tensor, gen_queries:List[str], tokenizer:T5Tokenizer) -> Dict[str, float]:
    """Compute rouge score for a batch
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