"""
Training Utility Functions
"""

from typing import List, Dict
import re
import pandas as pd
from tqdm import tqdm
from evaluate import load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sentence_transformers import  InputExample, datasets
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

def load_triplet_dataset_retriever(csv_path:List[str], test_size=0.2, val_size=0.2, batch_size=8):
    df  =   pd.concat([pd.read_csv(path, sep="\t") for path in csv_path])

    examples    =   []

    # Creating the Input Examples with format (anchor, positive, negative)
    for row in df.iterrows():
        examples.append(InputExample(texts=[row[1]['document'], row[1]['postive'], row[1]['negative']]))

    # Shuffling and creating train, test and val dataset
    train_examples, test_examples   =   train_test_split(shuffle(examples), test_size=test_size)
    test_examples, val_examples     =   train_test_split(test_examples, test_size=val_size)

    train_dataloader    =   datasets.NoDuplicatesDataLoader(train_examples, batch_size=batch_size)

    return train_dataloader, test_examples, val_examples