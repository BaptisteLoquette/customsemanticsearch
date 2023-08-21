"""
Utility functions for Preprocessing
"""

from typing import Dict
from torch import Tensor
import copy
from transformers import T5Tokenizer

def clean_and_format(sample:Dict[str, str]) -> Dict[str, str]:
    """Convert each column to ASCII and modify the string by adding a prefix to each column"""
    sample['query']     =   f"{sample['query'].encode('ascii', 'ignore').decode('ascii')}"
    sample['positive']  =   f"generate_positive_query: {sample['positive'].encode('ascii', 'ignore').decode('ascii')}"
    sample['negative']  =   f"generate_negative_query: {sample['negative'].encode('ascii', 'ignore').decode('ascii')}"
    
    return sample

def tokenize_dset_query_gen(sample:Dict[str, str], tokenizer:T5Tokenizer) -> Dict[str, Tensor]:
    """Tokenizes Dataset using T5-small tokenizer (SentencePiece)"""
    source  =   tokenizer.encode_plus(sample['document'], max_length=512,
                                    truncation=True, padding='max_length',
                                    return_tensors="pt")                # Truncating over 512
    target  =   tokenizer.encode_plus(sample['query'], max_length=96,
                                      truncation=True, padding='max_length',
                                      return_tensors="pt")
    
    sample['source_ids']            =   source['input_ids']
    sample['source_attention_mask'] =   source['attention_mask']
    sample['target_ids']            =   target['input_ids']
    sample['target_attention_mask'] =   target['attention_mask']
    labels              =   copy.deepcopy(target['input_ids'])
    labels[labels==0]   =   -100    # Replace <pad>'s idx (0) with -100 in order for t5 to not compute the loss on the paddings
    sample['labels']    =   labels

    return sample