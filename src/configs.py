from pathlib import Path
from src.utils.basic_utils import read_yaml


class Configs:
    def __init__(self, config_path: Path):
        self.config = read_yaml(config_path)

    def get_config(self):
        return self.config

    def get_model_config(self):
        return self.config["model"]

    def get_tokenizer_config(self):
        return self.config["tokenizer"]

    def get_training_config(self):
        return self.config["training"]

    def get_logging_config(self):
        return self.config["logging"]
