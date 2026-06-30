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
        else:
            self.pad_id = tokenizer.pad_token_id
            self.mask_id = tokenizer.mask_token_id
            self.cls_id = tokenizer.cls_token_id
            self.sep_id = tokenizer.sep_token_id
            self.special_ids = set(x for x in tokenizer.all_special_ids if x is not None)
            self.vocab_size = len(tokenizer)

    def torch_mask_tokens(self, inputs):
        """
        Prepare masked tokens inputs/labels for MLM: 80% MASK, 10% random, 10% original.
        inputs shape: (B, L)
        """
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for val in self.special_ids:
            if val is not None:
                special_tokens_mask |= (inputs == val)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # only calculate loss on masked tokens

        # 80% of the time, replace masked input tokens with mask_id
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.mask_id

        # 10% of the time, replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.special_ids), self.vocab_size, labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The remaining 10% of the time we keep the original token
        return inputs, labels

    def __call__(self, samples):
        """
        samples: list of (input_ids, segment_ids)
        """
        all_input_ids = [torch.tensor(s[0], dtype=torch.long) for s in samples]
        all_segment_ids = [torch.tensor(s[1], dtype=torch.long) for s in samples]
        all_attn_masks = [torch.ones(len(s[0]), dtype=torch.long) for s in samples]

        # Pad to longest sequence in batch
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.pad_id
        )
        segment_ids = pad_sequence(all_segment_ids, batch_first=True, padding_value=0)
        attn_mask = pad_sequence(all_attn_masks, batch_first=True, padding_value=0)

        # Vectorized dynamic masking
        input_ids, mlm_labels = self.torch_mask_tokens(input_ids)

        return input_ids, segment_ids, attn_mask, mlm_labels


class BertPreTrainDataset(Dataset):
    """
    Returns raw (unpadded, unmasked) single sentences.
    All masking and padding is handled by BertDataCollator.
    """

    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.n = len(sentences)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        input_ids = self.tokenizer.encode(sent)
        segment_ids = [0] * len(input_ids)
        return input_ids, segment_ids

