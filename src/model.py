# import torch
from torch import nn
from transformers import GPTNeoForCausalLM, GPT2Tokenizer


class GPT_NEO(nn.Module):
    def __init__(self, model_path, tokenizer_path, device):
        super(GPT_NEO, self).__init__()
        self.model = GPTNeoForCausalLM.from_pretrained(model_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        self.device = device

    def quantize(self, **kwargs):
        pass

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
