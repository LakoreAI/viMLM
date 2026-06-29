import math
from collections import defaultdict


class UnigramTokenizer:
    """
    Unigram Language Model: starts with a large vocab, then prunes tokens
    whose removal causes the least loss in corpus likelihood.
    Score(token) = log P(token) under the unigram LM.
    Viterbi decodes the best segmentation.
    """

    def __init__(self, vocab_size=200):
        self.vocab_size = vocab_size
        self.scores = {}  # token → log-prob
        self.token2id = {}
        self.id2token = {}

    def train(self, texts, prune_ratio=0.8, num_iterations=3):
        # Step 1: seed vocab with all substrings (up to len 6)
        substr_freq = defaultdict(int)
        all_words = []
        for text in texts:
            for word in text.strip().split():
                all_words.append(word)
                for i in range(len(word)):
                    for j in range(i + 1, min(i + 7, len(word) + 1)):
                        substr_freq[word[i:j]] += 1

        # Initialize scores as log-unigram freq
        total = sum(substr_freq.values())
        self.scores = {tok: math.log(freq / total) for tok, freq in substr_freq.items()}

        for iteration in range(num_iterations):
            # Step 2: E-step — segment all words with current scores (Viterbi)
            token_counts = defaultdict(float)
            total_loss = 0.0

            for word in all_words:
                seg, loss = self._viterbi(word)
                total_loss += loss
                for tok in seg:
                    token_counts[tok] += 1.0

            # Step 3: M-step — re-estimate log-probs
            total_c = sum(token_counts.values())
            self.scores = {
                tok: math.log(c / total_c) for tok, c in token_counts.items() if c > 0
            }

            # Step 4: Prune — remove tokens that cost least when dropped
            if iteration < num_iterations - 1:
                target = max(self.vocab_size, int(len(self.scores) * prune_ratio))
                if len(self.scores) > target:
                    # Keep top tokens by score (proxy for importance)
                    keep = sorted(self.scores, key=self.scores.get, reverse=True)[
                        :target
                    ]
                    self.scores = {k: self.scores[k] for k in keep}

            print(
                f"  iter {iteration + 1}: vocab={len(self.scores)}, loss={total_loss:.2f}"
            )

        # Finalize vocab
        specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        tokens = specials + sorted(self.scores.keys())
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {v: k for k, v in self.token2id.items()}

    def _viterbi(self, word):
        """Best segmentation of `word` under current log-prob scores."""
        n = len(word)
        best = [(-math.inf, [])] * (n + 1)
        best[0] = (0.0, [])

        for i in range(1, n + 1):
            for j in range(max(0, i - 6), i):
                sub = word[j:i]
                if sub in self.scores:
                    score = best[j][0] + self.scores[sub]
                    if score > best[i][0]:
                        best[i] = (score, best[j][1] + [sub])

        # Fallback: character split if no path found
        if best[n][0] == -math.inf:
            return list(word), -len(word) * 10.0
        return best[n][1], -best[n][0]

    def encode(self, text):
        ids = [self.token2id["[CLS]"]]
        unk = self.token2id["[UNK]"]
        for word in text.strip().split():
            seg, _ = self._viterbi(word)
            ids += [self.token2id.get(tok, unk) for tok in seg]
        ids.append(self.token2id["[SEP]"])
        return ids

    def decode(self, ids):
        specials = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
        return "".join(
            self.id2token.get(i, "")
            for i in ids
            if self.id2token.get(i) not in specials
        )
