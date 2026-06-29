from collections import defaultdict
import re


class BPETokenizer:
    """
    Byte-Pair Encoding: iteratively merges the most frequent adjacent pair.
    RoBERTa uses this over raw bytes (no [UNK] possible).
    """

    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.merges = {}  # (pair) → merged_token
        self.vocab = {}

    def _get_pairs(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge(self, vocab, pair):
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in vocab:
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def train(self, texts):
        # Initialize: split words into characters + </w> end marker
        vocab = defaultdict(int)
        for text in texts:
            for word in text.strip().split():
                vocab[" ".join(list(word)) + " </w>"] += 1

        self.vocab = dict(vocab)
        num_merges = self.vocab_size - len(set(" ".join(self.vocab.keys()).split()))

        for i in range(num_merges):
            pairs = self._get_pairs(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self._merge(self.vocab, best)
            self.merges[best] = "".join(best)
            if (i + 1) % 100 == 0:
                print(f"  merge {i + 1}: {best} → {''.join(best)}")

        # Build token → id
        all_tokens = set()
        for word in self.vocab:
            all_tokens.update(word.split())
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.token2id = {t: i for i, t in enumerate(specials + sorted(all_tokens))}
        self.id2token = {v: k for k, v in self.token2id.items()}

    def _tokenize_word(self, word):
        word = " ".join(list(word)) + " </w>"
        for pair, merged in self.merges.items():
            bigram = re.escape(" ".join(pair))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            word = pattern.sub(merged, word)
        return word.split()

    def encode(self, text):
        unk = self.token2id["[UNK]"]
        ids = [self.token2id["[CLS]"]]
        for word in text.strip().split():
            for tok in self._tokenize_word(word):
                ids.append(self.token2id.get(tok, unk))
        ids.append(self.token2id["[SEP]"])
        return ids

    def decode(self, ids):
        specials = set(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        tokens = [
            self.id2token.get(i, "[UNK]")
            for i in ids
            if self.id2token.get(i) not in specials
        ]
        return "".join(tokens).replace("</w>", " ").strip()
