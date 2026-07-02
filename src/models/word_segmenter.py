from underthesea import word_tokenize


def segment_words(text: str | list[str]):
    """Tokenize a Vietnamese text into words using underthesea."""
    if isinstance(text, str):
        return word_tokenize(text, format="text")
    else:
        return [word_tokenize(sentence, format="text") for sentence in text]
