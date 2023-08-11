"""
Training T5 model for query generation 
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from datasets import load_from_disk
from src.utils.infer_utils import generate_batch_queries
from src.utils.train_utils import compute_rouge_score

parser  =   argparse.ArgumentParser(
    prog="train_t5_genQ",
    description="Train T5-small to generate synthetic Queries from documents"
)
parser.add_argument('dset_path', help="Path of the dataset", type=str)
parser.add_argument('path_out', help="Output dir to save the model & tokenizer", type=str)
parser.add_argument('--batch_size', help="Batch Size for training", type=int, default=4, required=False)
parser.add_argument('--n_epochs', help="Number of epochs to train the model for", type=int, default=1, required=False)
parser.add_argument('--val_prop', help="Proportion of the validation dataset", type=float, default=0.1, required=False)
parser.add_argument('--pretrained_model_path', help="Path to pretrained T5's states to initialize the model with", type=str, default="t5-small", required=False)
parser.add_argument('--tokenizer_path', help="Path to model's tokenizer", type=str, default="t5-small", required=False)
parser.add_argument('--pos_and_neg', help="True if the user wants to train T5 on positive queries as well as hard negative queries generation", type=bool, default=False)


class T5GenQ_FineTuner(pl.LightningModule):
    """Pytorch Lightning T5 Fine Tuner"""
    def __init__(self, batch_size, model, tokenizer):
        super(T5GenQ_FineTuner, self).__init__()
        self.bs         =   batch_size
        self.model      =   model
        self.tokenizer  =   tokenizer

    def forward(self, input_ids, attention_mask=None, decoder_inputs_ids=None, decoder_attn_mask=None, labels=None):
        outputs =   self.model(
            input_ids=input_ids.squeeze(),
            attention_mask=attention_mask.squeeze(),
            decoder_attention_mask=decoder_attn_mask.squeeze(),
            labels=labels.squeeze())

        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs =   self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_attention_mask'],
            decoder_inputs_ids=batch['target_ids'],
            decoder_attn_mask=batch['target_attention_mask'],
            labels=batch['labels']
        )

        loss    =   outputs[0]
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs =   self.forward(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_attention_mask'],
            decoder_inputs_ids=batch['target_ids'],
            decoder_attn_mask=batch['target_attention_mask'],
            labels=batch['labels']
        )
        generated_Q =   generate_batch_queries(batch, model, tokenizer) # Generates queries and decode target input ids in order to compute ROUGE scores
        rouge_score =   compute_rouge_score(batch['target_ids'], generated_Q, tokenizer) # Computes ROUGE scores

        loss    =   outputs[0]
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_ROUGE', rouge_score['rougeLsum'], prog_bar=True, on_epoch=True)

        return loss
    
    def train_dataloader(self):
        return DataLoader(dset['train'], batch_size=self.bs, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(dset['test'], batch_size=self.bs, num_workers=4)
    
    def configure_optimizers(self):
        optimizer   =   AdamW(self.parameters(), lr=3e-4, eps=1e-8)
        return optimizer


if "__main__" == __name__:
    args                    =   parser.parse_args()
    dset_path               =   args.dset_path
    batch_size              =   args.batch_size
    val_prop                =   args.val_prop
    pretrained_model_path   =   args.pretrained_model_path
    tokenizer_path          =   args.tokenizer_path
    n_epochs                =   args.n_epochs
    path_out                =   args.path_out
    pos_and_neg             =   args.pos_and_neg

    device      =   "gpu" if torch.cuda.is_available() else "cpu"

    tokenizer   =   T5Tokenizer.from_pretrained(pretrained_model_path)  # Load Pretrained model
    model       =   T5ForConditionalGeneration.from_pretrained(tokenizer_path)   # Load model's Tokenizer

    dset        =   load_from_disk(dset_path).with_format('torch')  # Load Preprocessed Dataset
    dset        =   dset.train_test_split(test_size=val_prop)            # Split dataset

    torch.cuda.empty_cache()
    finetune_model  =   T5GenQ_FineTuner(batch_size, model, tokenizer)
    trainer =   pl.Trainer(max_epochs=n_epochs, devices=1, accelerator=device)
    trainer.fit(finetune_model)

    finetune_model.model.save_pretrained(os.path.join(path_out, "model"))
    tokenizer.save_pretrained(os.path.join(path_out, "tokenizer"))