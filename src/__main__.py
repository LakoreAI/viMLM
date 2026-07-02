import argparse

import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from src.utils.basic_utils import get_run_name
from src.utils.data_utils import load_sentences_from_file
from src.utils.model_utils import count_parameters, seed_everything
from src.models.config import Config
from src.models.bert import BertForPreTraining
from src.dataset import BertDataCollator, BertPreTrainDataset
from src.pipe.training_pipe import train
from src.pipe.eval_pipe import eval
from src.callbacks.wandb_callbacks import WandbCallback
from src.callbacks.checkpoint_callback import CheckpointCallback


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
    seed_everything(getattr(cfg, "seed", 42))

    # 1) Data Loading
    print("Loading corpus...")
    sentences = load_sentences_from_file(args.data_path)
    print(f"Loaded {len(sentences)} sentences.")

    use_wwm = getattr(cfg, "use_wwm", False)
    if use_wwm:
        print("Segmenting Vietnamese words for Whole Word Masking...")
        from src.models.word_segmenter import segment_words

        sentences = segment_words(sentences)
        print("Segmentation complete.")

    # 2) Split Train/Eval
    dataset = BertPreTrainDataset(sentences, cfg.tokenizer, use_wwm=use_wwm)
    eval_len = int(len(dataset) * args.eval_ratio)
    train_dataset, eval_dataset = random_split(
        dataset, [len(dataset) - eval_len, eval_len]
    )

    # 3) Data Collator
    mlm_probability = getattr(cfg, "mlm_probability", 0.15)
    data_collator = BertDataCollator(cfg.tokenizer, mlm_probability=mlm_probability)

    # 4) Data Loaders
    batch_size = getattr(cfg, "batch_size", args.batch_size)
    eval_batch_size = getattr(cfg, "eval_batch_size", args.eval_batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
    )

    # 5) Model Initialization
    model = BertForPreTraining(cfg).to(device)
    count_parameters(model)

    # 6) Optimizer & Scheduler
    # Prefer YAML config values; fall back to CLI args for backward compatibility
    lr = getattr(cfg, "learning_rate", args.lr)
    weight_decay = getattr(cfg, "weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = getattr(cfg, "epochs", args.epochs)
    grad_accum_steps = getattr(cfg, "gradient_accumulation_steps", 1)

    # Calculate optimization steps
    steps_per_epoch = len(train_loader) // grad_accum_steps
    if len(train_loader) % grad_accum_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * epochs

    warmup_ratio = getattr(cfg, "warmup_ratio", None)
    warmup_steps = (
        int(total_steps * warmup_ratio) if warmup_ratio else args.warmup_steps
    )
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # 7) Train
    callbacks = []
    wandb_cb = None
    if cfg.use_wandb:
        _wandb_run_name = get_run_name(
            model_name=cfg.wandb_run_name, experiment_type=cfg.task
        )
        wandb_cb = WandbCallback(
            project_name=cfg.wandb_project,
            run_name=_wandb_run_name,
            config=cfg.raw_config,
            entity=cfg.wandb_entity,
        )

    # Checkpoint callback integration
    checkpoint_cb = CheckpointCallback(
        model=model,
        checkpoint_dir=getattr(cfg, "checkpoint_dir", "./checkpoints"),
        save_best=getattr(cfg, "save_best", True),
        save_last=getattr(cfg, "save_last", True),
        save_steps=getattr(cfg, "save_steps", None),
        wandb_callback=wandb_cb,
    )
    callbacks.append(checkpoint_cb)

    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    eval_steps = getattr(cfg, "eval_steps", None) or args.eval_steps

    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=epochs,
        eval_loader=eval_loader,
        callbacks=callbacks,
        gradient_accumulation_steps=grad_accum_steps,
        eval_steps=eval_steps,
    )


if __name__ == "__main__":
    main()
