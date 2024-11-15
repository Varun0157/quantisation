import os
import torch
import torch.nn as nn

from transformers import GPTNeoForCausalLM, GPT2Tokenizer


class GPTNeo(nn.Module):
    def __init__(self, models_path: str, device: torch.device):
        super().__init__()
        self.device = device

        tok_path = os.path.join(models_path, "gpt-neo_tokenizer")
        mod_path = os.path.join(models_path, "gpt-neo_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPTNeoForCausalLM.from_pretrained(mod_path)
        self.model.to(self.device)  # type: ignore
        self.model.eval()

    def quantize(self):
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    def forward(self, prompt: str):
        encodings = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        NEW_MIN, NEW_MAX = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        OLD_MIN, OLD_MAX = tensor.min(), tensor.max()

        VAL_RANGE = NEW_MAX - NEW_MIN
        scale = (OLD_MAX - OLD_MIN) / VAL_RANGE
        zero_point = OLD_MIN

        quantized = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)
        return quantized
