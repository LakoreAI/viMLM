import argparse
import pickle
from pathlib import Path

from src.utils.data_utils import load_sentences_from_file
from src.models.tokenizers import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom tokenizer.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/wikipedia_corpus.txt",
        help="Path to the training corpus text file.",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="bpe",
        choices=["bpe", "wp", "unigram", "char"],
        help="Type of tokenizer to build.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Vocabulary size of the tokenizer.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="checkpoints/tokenizer.pkl",
        help="Path to save the trained tokenizer.",
    )
    return parser.parse_args()


def build_tokenizer(data_path, tokenizer_type, vocab_size, save_path):
    print(f"Loading corpus from {data_path}...")
    sentences = load_sentences_from_file(data_path)
    print(f"Loaded {len(sentences)} sentences for tokenizer training.")

    print(f"Initializing {tokenizer_type} tokenizer with vocab size {vocab_size}...")
    tokenizer = get_tokenizer({"type": tokenizer_type, "vocab_size": vocab_size})

    print("Training tokenizer...")
    tokenizer.train(sentences)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Tokenizer trained and successfully saved to {save_path}")
    print(f"Trained vocabulary size: {len(tokenizer.token2id)}")


def main():
    args = parse_args()
    build_tokenizer(
        data_path=args.data_path,
        tokenizer_type=args.type,
        vocab_size=args.vocab_size,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
