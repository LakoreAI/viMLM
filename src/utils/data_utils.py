def load_sentences_from_file(path: str) -> list[str]:
    """
    Load sentences from a text file where each line is a sentence.
    """
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences
