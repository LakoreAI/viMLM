from pathlib import Path
from src.utils.basic_utils import read_yaml
from src.models.tokenizers import get_tokenizer


class Config:
    def __init__(self, config_path: str):
        config_path = Path(config_path)
        self.raw_config = read_yaml(config_path)

        # Model config
        model_cfg = self.raw_config.get("model", {})
        self.pad_token_id = model_cfg.get("pad_token_id", 0)
        self.vocab_size = model_cfg.get("vocab_size", 1000)
        self.max_seq_len = model_cfg.get("max_seq_len", 128)
        self.hidden_size = model_cfg.get("hidden_size", 256)
        self.num_layers = model_cfg.get("num_layers", 4)
        self.num_heads = model_cfg.get("num_heads", 4)
        self.ff_dim = model_cfg.get("ff_dim", 1024)
        self.dropout = model_cfg.get("dropout", 0.1)
        self.use_rope = model_cfg.get("use_rope", False)

        # Tokenizer config
        tokenizer_cfg = self.raw_config.get("tokenizer", {})
        self.tokenizer_name = tokenizer_cfg.get("name", "bert-base-uncased")

        # Instantiate tokenizer
        if "type" in tokenizer_cfg:
            self.tokenizer = get_tokenizer(tokenizer_cfg)
            if hasattr(self.tokenizer, "vocab_size"):
                self.vocab_size = self.tokenizer.vocab_size
            elif hasattr(self.tokenizer, "token2id"):
                self.vocab_size = len(self.tokenizer.token2id)
            else:
                self.vocab_size = model_cfg.get("vocab_size", 1000)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.vocab_size = len(self.tokenizer)

