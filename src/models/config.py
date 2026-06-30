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
        self.use_unet_shrink = model_cfg.get("use_unet_shrink", False)
        self.unet_bottleneck_ratio = model_cfg.get("unet_bottleneck_ratio", 0.5)
        self.layer_hidden_sizes = model_cfg.get("layer_hidden_sizes", None)

        # Training/wandb config
        training_cfg = self.raw_config.get("training", {})
        self.use_wandb = training_cfg.get("use_wandb", False)
        self.wandb_project = training_cfg.get("wandb_project", "viMLM")
        self.wandb_entity = training_cfg.get("wandb_entity", None)
        self.wandb_run_name = training_cfg.get("wandb_run_name", "viMLM")
        self.learning_rate = training_cfg.get("learning_rate", 1e-4)
        self.weight_decay = training_cfg.get("weight_decay", 0.01)
        self.warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
        self.seed = training_cfg.get("seed", 42)
        self.eval_steps = training_cfg.get("eval_steps", None)
        self.save_steps = training_cfg.get("save_steps", None)
        self.save_best = training_cfg.get("save_best", False)
        self.save_last = training_cfg.get("save_last", True)
        self.checkpoint_dir = training_cfg.get("checkpoint_dir", "./checkpoints")

        # Tokenizer config
        tokenizer_cfg = self.raw_config.get("tokenizer", {})
        self.tokenizer_name = tokenizer_cfg.get("name", "bert-base-uncased")

        # Instantiate tokenizer
        if "type" in tokenizer_cfg:
            import pickle

            vocab_path = tokenizer_cfg.get("vocab_path")
            if vocab_path and Path(vocab_path).exists():
                with open(vocab_path, "rb") as f:
                    self.tokenizer = pickle.load(f)
            else:
                self.tokenizer = get_tokenizer(tokenizer_cfg)

            if hasattr(self.tokenizer, "token2id"):
                self.vocab_size = len(self.tokenizer.token2id)
            elif hasattr(self.tokenizer, "vocab_size"):
                self.vocab_size = self.tokenizer.vocab_size
            else:
                self.vocab_size = model_cfg.get("vocab_size", 1000)
        else:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.vocab_size = len(self.tokenizer)
