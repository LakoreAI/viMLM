import argparse

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from src.utils.data_utils import load_sentences_from_file
from src.utils.model_utils import count_parameters
from src.models.config import Config
from src.models.bert import BertForPreTraining
from src.dataset import BertDataCollator, BertPreTrainDataset
from src.pipe.training_pipe import train
from src.pipe.eval_pipe import eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="data/raw/wikipedia_corpus.txt"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--eval-steps", type=int, default=5000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--config-path", type=str, default="config/training_config.yml")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["linear", "cosine"]
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(args.config_path)

    # 1) Data Loading
    print("Loading corpus...")
    sentences = load_sentences_from_file(args.data_path)
    print(f"Loaded {len(sentences)} sentences.")

    # 2) Split Train/Eval
    dataset = BertPreTrainDataset(sentences, cfg.tokenizer)
    eval_len = int(len(dataset) * args.eval_ratio)
    train_dataset, eval_dataset = random_split(
        dataset, [len(dataset) - eval_len, eval_len]
    )

    # 3) Data Collator
    data_collator = BertDataCollator(cfg.tokenizer, mlm_probability=0.15)

    # 4) Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
    )

    # 5) Model Initialization
    model = BertForPreTraining(cfg).to(device)
    count_parameters(model)

    # 6) Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )

    # 7) Train
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        eval_loader=eval_loader,
    )


if __name__ == "__main__":
    main()

