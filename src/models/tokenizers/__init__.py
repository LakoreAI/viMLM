from src.models.tokenizers.bpe_tokenizer import BPETokenizer
from src.models.tokenizers.wordpiece_tokenizer import WordPieceTokenizer
from src.models.tokenizers.unigram_tokenizer import UnigramTokenizer
from src.models.tokenizers.char_tokenizer import CharTokenizer


def get_tokenizer(tokenizer_config: dict) -> object:
    """
    Returns a tokenizer instance based on the provided configuration.
    """
    tokenizer_type = tokenizer_config.get("type")
    vocab_size = tokenizer_config.get("vocab_size")

    if tokenizer_type == "bpe":
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        return BPETokenizer(**kwargs)
    elif tokenizer_type == "wp":
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        return WordPieceTokenizer(**kwargs)
    elif tokenizer_type == "unigram":
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        return UnigramTokenizer(**kwargs)
    elif tokenizer_type == "char":
        return CharTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


__all__ = [
    "BPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "CharTokenizer",
    "get_tokenizer",
]
