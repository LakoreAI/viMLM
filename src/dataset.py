import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class BertDataCollator:
    """
    Collates raw sentences into a padded batch with dynamic MLM masking.

    Responsibilities:
      - Pads input_ids and segment_ids to the longest sequence in the batch
      - Creates attention_mask (1 = real token, 0 = pad)
      - Applies MLM masking on-the-fly (80/10/10 rule)

    Why dynamic masking here instead of in Dataset.__getitem__?
      Each epoch sees a different mask for the same sample →
      effectively multiplies your data diversity for free.
    """

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        if hasattr(tokenizer, "SPECIAL"):
            self.pad_id = tokenizer.SPECIAL["[PAD]"]
            self.mask_id = tokenizer.SPECIAL["[MASK]"]
            self.cls_id = tokenizer.SPECIAL["[CLS]"]
            self.sep_id = tokenizer.SPECIAL["[SEP]"]
            self.special_ids = set(tokenizer.SPECIAL.values())
            self.vocab_size = len(tokenizer.token2id)
        elif hasattr(tokenizer, "token2id"):
            self.pad_id = tokenizer.token2id["[PAD]"]
            self.mask_id = tokenizer.token2id["[MASK]"]
            self.cls_id = tokenizer.token2id["[CLS]"]
            self.sep_id = tokenizer.token2id["[SEP]"]
            self.special_ids = set(
                tokenizer.token2id[k]
                for k in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
                if k in tokenizer.token2id
            )
            self.vocab_size = len(tokenizer.token2id)
        else:
            self.pad_id = tokenizer.pad_token_id
            self.mask_id = tokenizer.mask_token_id
            self.cls_id = tokenizer.cls_token_id
            self.sep_id = tokenizer.sep_token_id
            self.special_ids = set(
                x for x in tokenizer.all_special_ids if x is not None
            )
            self.vocab_size = len(tokenizer)

        non_special_list = [
            i for i in range(self.vocab_size) if i not in self.special_ids
        ]
        if not non_special_list:
            non_special_list = [0]
        self.non_special_ids = torch.tensor(non_special_list, dtype=torch.long)

    def torch_mask_tokens(self, inputs, word_ids=None):
        """
        Prepare masked tokens inputs/labels for MLM: 80% MASK, 10% random, 10% original.
        inputs shape: (B, L)
        """
        labels = inputs.clone()

        if word_ids is None:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)

            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
            for val in self.special_ids:
                if val is not None:
                    special_tokens_mask |= inputs == val

            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
        else:
            # Whole Word Masking (WWM)
            masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
            for batch_idx in range(inputs.shape[0]):
                seq_word_ids = word_ids[batch_idx]
                unique_words = set(seq_word_ids[seq_word_ids != -1].tolist())
                if not unique_words:
                    continue
                num_to_mask = max(1, round(len(unique_words) * self.mlm_probability))
                words_to_mask = set(random.sample(list(unique_words), num_to_mask))

                for token_idx, w_id in enumerate(seq_word_ids.tolist()):
                    if w_id in words_to_mask:
                        masked_indices[batch_idx, token_idx] = True

        labels[~masked_indices] = -100  # only calculate loss on masked tokens

        # 80% of the time, replace masked input tokens with mask_id
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.mask_id

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_indices = torch.randint(
            0,
            len(self.non_special_ids),
            labels.shape,
            dtype=torch.long,
            device=inputs.device,
        )
        random_words = self.non_special_ids.to(inputs.device)[random_indices]
        inputs[indices_random] = random_words[indices_random]

        # The remaining 10% of the time we keep the original token
        return inputs, labels

    def __call__(self, samples):
        """
        samples: list of (input_ids, segment_ids, word_ids)
        """
        all_input_ids = [torch.tensor(s[0], dtype=torch.long) for s in samples]
        all_segment_ids = [torch.tensor(s[1], dtype=torch.long) for s in samples]
        all_attn_masks = [torch.ones(len(s[0]), dtype=torch.long) for s in samples]

        if len(samples[0]) > 2:
            all_word_ids = [torch.tensor(s[2], dtype=torch.long) for s in samples]
            word_ids = pad_sequence(all_word_ids, batch_first=True, padding_value=-1)
        else:
            word_ids = None

        # Pad to longest sequence in batch
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.pad_id
        )
        segment_ids = pad_sequence(all_segment_ids, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(all_attn_masks, batch_first=True, padding_value=0)

        # Vectorized dynamic masking
        input_ids, mlm_labels = self.torch_mask_tokens(input_ids, word_ids=word_ids)

        return input_ids, segment_ids, attn_mask, mlm_labels


class BertPreTrainDataset(Dataset):
    """
    Returns raw (unpadded, unmasked) single sentences.
    All masking and padding is handled by BertDataCollator.
    Supports Whole Word Masking (WWM) word_ids creation.
    """

    def __init__(self, sentences, tokenizer, use_wwm=False):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.use_wwm = use_wwm
        self.n = len(sentences)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sent = self.sentences[idx]

        if self.use_wwm:
            # Under_score linked word list
            words = sent.split()
            # If the tokenizer is HF tokenizer supporting is_split_into_words
            if hasattr(self.tokenizer, "is_fast") or hasattr(
                self.tokenizer, "word_ids"
            ):
                encoding = self.tokenizer(
                    words,
                    is_split_into_words=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=512,
                )
                input_ids = encoding["input_ids"]
                word_ids = encoding.word_ids()
                word_ids = [w if w is not None else -1 for w in word_ids]
            else:
                # Custom tokenizer fallback: treat each token as its own word
                input_ids = self.tokenizer.encode(sent)
                word_ids = list(range(len(input_ids)))
        else:
            input_ids = self.tokenizer.encode(sent)
            word_ids = list(range(len(input_ids)))

        segment_ids = [0] * len(input_ids)
        return input_ids, segment_ids, word_ids
