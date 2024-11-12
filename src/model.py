import os
import torch
import torch.nn as nn

from transformers import GPTNeoForCausalLM, GPT2Tokenizer


class GPTNeo(nn.Module):
    def __init__(self, models_path: str, device):
        super().__init__()
        self.device = device

        tok_path = os.path.join(models_path, "gpt-neo_tokenizer")
        mod_path = os.path.join(models_path, "gpt-neo_model")
        self.tokenizer = GPT2Tokenizer.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPTNeoForCausalLM.from_pretrained(mod_path).to(device)
        self.model.eval()

    def quantize(self):
        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.int8
        )

    def forward(self, prompt: str):
        encodings = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=True
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        return outputs
