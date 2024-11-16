import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.from_scratch import quantize_linear_layers


class AutoModel(nn.Module):
    def __init__(self, models_path: str, device: torch.device, model_alias: str):
        super().__init__()
        self.device = device

        self.tok_path = os.path.join(models_path, model_alias + "_tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.mod_path = os.path.join(models_path, model_alias + "_model")
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            self.mod_path
        )
        self.model.to(self.device)  # type: ignore

    def quantize_custom(self, goal_dtype: torch.dtype = torch.int8):
        quantize_linear_layers(self.model, goal_dtype)

    def memory_footprint(self):
        return self.model.get_memory_footprint()  # type: ignore

    def forward(self, sentences: str):
        encodings = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = encodings["input_ids"].to(self.device)  # type: ignore
        attention_mask = encodings["attention_mask"].to(self.device)  # type: ignore

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)  # type: ignore
        return outputs
