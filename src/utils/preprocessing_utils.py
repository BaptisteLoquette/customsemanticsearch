"""
Utility functions for Preprocessing
"""

from typing import Dict
from torch import Tensor
import copy
import nltk
import random
from tqdm import tqdm
from sentence_transformers import InputExample
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

def preprocess_squad_for_contextual_compression(dset):
    examples    =   []

    for example in tqdm(dset):
        context     =   example['context']
        question    =   example['question']
        answers     =   example['answers']
        if len(answers['answer_start']) == 0:   # A certain number of samples do not contain any answer, thus we ignore those
            continue

        # Retrieve the sentence
        sentences   =   nltk.sent_tokenize(context)
        sent_len    =   0
        prev_len    =   0

        for i_sent in range(len(sentences)):
            sent_len    +=   len(sentences[i_sent])
            if sent_len >= answers['answer_start'][0] and prev_len < answers['answer_start'][0]:
                break
            prev_len    =   sent_len
        
        pos_sent    =   sentences[i_sent]

        # Sample a negative sentence
        neg_idxs    =   list(range(len(sentences)))
        neg_idxs.pop(i_sent)
        examples.append(InputExample(texts=[question, pos_sent], label=1))
        if len(neg_idxs) > 0:
            neg_sent    =   sentences[random.choice(neg_idxs)]
            examples.append(InputExample(texts=[question, neg_sent], label=0))
        else:   # If the context contains only one sentence (i.e. the positive one) we do not append a negative example
            continue

    return examples