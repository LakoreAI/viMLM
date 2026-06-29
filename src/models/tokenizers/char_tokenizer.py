class CharTokenizer:
    """
    Character-level tokenizer.
    Every character is a token — no subword splitting.
    Vocab = all unique characters in the corpus + 5 special tokens.

    Pros: zero [UNK] on known languages, trivial to implement
    Cons: very long sequences (512 chars = 512 tokens), loses morphology signal
    """

    SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}

    def __init__(self):
        self.token2id = dict(self.SPECIAL)
        self.id2token = {v: k for k, v in self.SPECIAL.items()}

    def train(self, texts, max_vocab=1000):
        freq = {}
        for text in texts:
            for ch in text:
                freq[ch] = freq.get(ch, 0) + 1
        # Add characters sorted by frequency
        for ch, _ in sorted(freq.items(), key=lambda x: -x[1]):
            if len(self.token2id) >= max_vocab:
                break
            if ch not in self.token2id:
                idx = len(self.token2id)
                self.token2id[ch] = idx
                self.id2token[idx] = ch

    def encode(self, text, max_len=None):
        unk = self.SPECIAL["[UNK]"]
        ids = [self.SPECIAL["[CLS]"]]
        for ch in text:
            ids.append(self.token2id.get(ch, unk))
        ids.append(self.SPECIAL["[SEP]"])
        if max_len:
            ids = ids[:max_len]
            ids += [self.SPECIAL["[PAD]"]] * max(0, max_len - len(ids))
        return ids

    def decode(self, ids):
        specials = set(self.SPECIAL.values())
        return "".join(self.id2token.get(i, "[UNK]") for i in ids if i not in specials)

    @property
    def vocab_size(self):
        return len(self.token2id)
