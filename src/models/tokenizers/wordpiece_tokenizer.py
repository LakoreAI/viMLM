from collections import defaultdict


class WordPieceTokenizer:
    """
    WordPiece: like BPE but merges the pair that maximizes
    score = freq(AB) / (freq(A) * freq(B))
    This prefers rare subwords — opposite bias to BPE.
    Subwords that are not word-initial get '##' prefix.
    """

    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}

    def train(self, texts):
        # Count word frequencies
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.strip().split():
                word_freq[word] += 1

        # Initialize vocab: chars (first char plain, rest with ##)
        vocab = {}
        for word, freq in word_freq.items():
            chars = [word[0]] + ["##" + c for c in word[1:]]
            vocab[" ".join(chars)] = freq

        # Collect initial character-level tokens
        char_vocab = set()
        for word in vocab:
            char_vocab.update(word.split())

        num_merges = self.vocab_size - len(char_vocab) - 5  # reserve for specials

        for step in range(max(0, num_merges)):
            # Score each pair: freq(AB) / freq(A) / freq(B)
            pair_freq = defaultdict(int)
            token_freq = defaultdict(int)

            for word, freq in vocab.items():
                symbols = word.split()
                for sym in symbols:
                    token_freq[sym] += freq
                for i in range(len(symbols) - 1):
                    pair_freq[(symbols[i], symbols[i + 1])] += freq

            if not pair_freq:
                break

            scores = {
                pair: freq / (token_freq[pair[0]] * token_freq[pair[1]])
                for pair, freq in pair_freq.items()
            }
            best = max(scores, key=scores.get)
            merged = best[0] + best[1].lstrip("#")  # ##bc → bc when merged

            # Apply merge
            new_vocab = {}
            import re

            pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(best)) + r"(?!\S)")
            for word in vocab:
                new_vocab[pattern.sub(merged, word)] = vocab[word]
            vocab = new_vocab

            if (step + 1) % 100 == 0:
                print(f"  merge {step + 1}: {best[0]} + {best[1]} → {merged}")

        # Build token2id
        all_toks = set()
        for word in vocab:
            all_toks.update(word.split())
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.token2id = {t: i for i, t in enumerate(specials + sorted(all_toks))}
        self.id2token = {v: k for k, v in self.token2id.items()}

    def encode(self, text, max_len=None):
        unk = self.token2id.get("[UNK]", 1)
        ids = [self.token2id["[CLS]"]]
        for word in text.strip().split():
            ids += self._tokenize_word(word, unk)
        ids.append(self.token2id["[SEP]"])
        if max_len:
            ids = ids[:max_len] + [self.token2id["[PAD]"]] * max(0, max_len - len(ids))
        return ids

    def _tokenize_word(self, word, unk_id):
        """Greedy longest-match from left."""
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = None
            prefix = "" if start == 0 else "##"
            while start < end:
                substr = prefix + word[start:end]
                if substr in self.token2id:
                    found = substr
                    break
                end -= 1
            if found is None:
                return [unk_id]  # whole word → [UNK]
            tokens.append(self.token2id[found])
            start = end
        return tokens

    def decode(self, ids):
        specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        out, buf = [], ""
        for i in ids:
            tok = self.id2token.get(i, "")
            if tok in specials:
                continue
            if tok.startswith("##"):
                buf += tok[2:]
            else:
                if buf:
                    out.append(buf)
                buf = tok
        if buf:
            out.append(buf)
        return " ".join(out)
