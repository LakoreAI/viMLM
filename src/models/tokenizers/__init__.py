from src.models.tokenizers.bpe_tokenizer import BPETokenizer
from src.models.tokenizers.wordpiece_tokenizer import WordPieceTokenizer
from src.models.tokenizers.unigram_tokenizer import UnigramTokenizer
from src.models.tokenizers.char_tokenizer import CharTokenizer


def get_tokenizer(tokenizer_config: dict) -> object:
    """
    Returns a tokenizer instance based on the provided configuration.
    """
    tokenizer_type = tokenizer_config.get("type")
    if tokenizer_type == "bpe":
        return BPETokenizer(**tokenizer_config)
    elif tokenizer_type == "wp":
        return WordPieceTokenizer(**tokenizer_config)
    elif tokenizer_type == "unigram":
        return UnigramTokenizer(**tokenizer_config)
    elif tokenizer_type == "char":
        return CharTokenizer(**tokenizer_config)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


__all__ = [
    "BPETokenizer",
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "CharTokenizer",
    "get_tokenizer",
]
