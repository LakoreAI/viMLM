import torch
from torch.nn.utils.rnn import pad_sequence


class BertDataCollator:
    """
    Collates raw sentence pairs into a padded batch with dynamic MLM masking.

    Responsibilities:
      - Pads input_ids and segment_ids to the longest sequence in the batch
      - Creates attention_mask (1 = real token, 0 = pad)
      - Applies MLM masking on-the-fly (80/10/10 rule)
      - Returns nsp_labels as a tensor

    Why dynamic masking here instead of in Dataset.__getitem__?
      Each epoch sees a different mask for the same sample →
      effectively multiplies your data diversity for free.
    """

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_id = tokenizer.SPECIAL["[PAD]"]
        self.mask_id = tokenizer.SPECIAL["[MASK]"]
        self.cls_id = tokenizer.SPECIAL["[CLS]"]
        self.sep_id = tokenizer.SPECIAL["[SEP]"]
        self.special_ids = set(tokenizer.SPECIAL.values())

    def _mask_tokens(self, input_ids):
        """
        Apply MLM masking to a single sequence (list of ints).
        Returns (masked_ids, labels) both as lists.
          labels[i] = original token if masked, else -100
        """
        labels = [-100] * len(input_ids)
        masked = input_ids[:]

        for i, tok in enumerate(input_ids):
            if tok in self.special_ids:  # never mask special tokens
                continue
            if torch.rand(1).item() < self.mlm_probability:
                labels[i] = tok
                r = torch.rand(1).item()
                if r < 0.80:
                    masked[i] = self.mask_id
                elif r < 0.90:
                    masked[i] = torch.randint(
                        len(self.special_ids), len(self.tokenizer.token2id), (1,)
                    ).item()
                # else: keep original (10%)

        return masked, labels

    def __call__(self, samples):
        """
        samples: list of (input_ids, segment_ids, nsp_label)
                 input_ids and segment_ids are plain Python lists (unpadded).
        """
        all_input_ids = []
        all_segment_ids = []
        all_mlm_labels = []
        all_attn_masks = []
        all_nsp_labels = []

        for input_ids, segment_ids, nsp_label in samples:
            masked_ids, mlm_labels = self._mask_tokens(input_ids)

            all_input_ids.append(torch.tensor(masked_ids, dtype=torch.long))
            all_segment_ids.append(torch.tensor(segment_ids, dtype=torch.long))
            all_mlm_labels.append(torch.tensor(mlm_labels, dtype=torch.long))
            all_attn_masks.append(torch.ones(len(input_ids), dtype=torch.long))
            all_nsp_labels.append(nsp_label)

        # Pad to longest sequence in batch
        input_ids = pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.pad_id
        )
        segment_ids = pad_sequence(all_segment_ids, batch_first=True, padding_value=0)
        mlm_labels = pad_sequence(all_mlm_labels, batch_first=True, padding_value=-100)
        attn_mask = pad_sequence(all_attn_masks, batch_first=True, padding_value=0)
        nsp_labels = torch.tensor(all_nsp_labels, dtype=torch.long)

        return input_ids, segment_ids, attn_mask, mlm_labels, nsp_labels


class BertPreTrainDataset(Dataset):
    """
    Returns raw (unpadded, unmasked) sentence pairs.
    All masking and padding is handled by BertDataCollator.
    """

    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.n = len(sentences)

    def __len__(self):
        return self.n - 1

    def __getitem__(self, idx):
        sent_a = self.sentences[idx]
        if torch.rand(1).item() < 0.5:
            sent_b, nsp_label = self.sentences[idx + 1], 0  # IsNext
        else:
            sent_b, nsp_label = self.sentences[torch.randint(0, self.n, (1,)).item()], 1

        input_ids, segment_ids = self.tokenizer.encode_pair(sent_a, sent_b)
        return input_ids, segment_ids, nsp_label  # plain lists, no tensors yet
