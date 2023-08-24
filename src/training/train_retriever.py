import os
import torch
import wandb
import argparse
from sentence_transformers import SentenceTransformer, losses
from src.utils.train_utils import load_triplet_dataset_retriever
from sentence_transformers.evaluation import TripletEvaluator

parser  =   argparse.ArgumentParser(
    prog="train_retriever",
    description="Trains Retriever on (Queries, Documents) pairs dataset"
)

parser.add_argument('path_out', help="Output path to save the CSV generated dataset", type=str)
parser.add_argument('--csv_path', action='append', help="List of CSV datasets Input paths", required=True)
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warmup_prop', type=float, default=0.1)
parser.add_argument('--test_size', type=float, default=0.2)
parser.add_argument('--val_size', type=float, default=0.2)
parser.add_argument('--pretrained_model_path', help="Path to pretrained T5's states to initialize the model with", type=str, default='msmarco-distilbert-base-dot-prod-v3')
parser.add_argument('--wandb_project', type=str, default="Retriever-GenQ")

def callback_model(score, epoch, steps):
    wandb.log({"train/epoch" : epoch,
                "train/steps": steps,
                "val/score" : score})
    
if "__main__" == __name__:
    args                    =   parser.parse_args()
    csv_path                =   args.csv_path
    path_out                =   args.path_out
    num_epochs              =   args.num_epochs
    batch_size              =   args.batch_size
    warmup_prop             =   args.warmup_prop
    test_size               =   args.test_size
    val_size                =   args.val_size
    pretrained_model_path   =   args.pretrained_model_path
    wandb_project           =   args.wandb_project

    wandb.init(project=wandb_project)

    device          =   torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dloader, test_examples, val_examples  =   load_triplet_dataset_retriever(csv_path, test_size=test_size, val_size=val_size, batch_size=batch_size)
    val_evaluator   =   TripletEvaluator.from_input_examples(val_examples)
    test_evaluator  =   TripletEvaluator.from_input_examples(test_examples)

    model           =   SentenceTransformer(pretrained_model_path).to(device)
    loss            =   losses.TripletLoss(model)
    warmup_steps    =   int(len(train_dloader) * num_epochs * warmup_prop)

    model.fit(train_objectives=[(train_dloader, loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            evaluator=val_evaluator,
            evaluation_steps=1000,
            callback=callback_model)
    model.save(path_out)

    wandb.log({"test/score" : model.evaluate(test_evaluator)})