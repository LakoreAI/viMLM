def load_sentences_from_file(path: str) -> list[str]:
    """
    Load sentences from a text file where each line is a sentence.
    Exact duplicate lines are dropped (order-preserving) to avoid wasting
    compute and reinforcing memorization on repeated text.
    """
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    seen = set()
    deduped = [s for s in sentences if not (s in seen or seen.add(s))]

    n_dupes = len(sentences) - len(deduped)
    if n_dupes:
        print(
            f"Removed {n_dupes} duplicate sentence(s) ({n_dupes / len(sentences):.1%} of corpus)"
        )

    return deduped
