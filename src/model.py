import os
import logging
from typing import List
from enum import Enum

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.from_scratch import quantize_layers


class QuantisationType(Enum):
    none = "none"
    custom_whole = "custom_whole"
    custom_selective = "custom_selective"
    bnb_4 = "bnb_4"
    bnb_8 = "bnb_8"
    bnb_4_nf4 = "bnb_4_nf4"
    bnb_8_nf4 = "bnb_8_nf4"


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
        self.model.eval()  # type: ignore

    def quantize_custom(
        self,
        goal_dtype: torch.dtype = torch.int8,
        select_layers: List[str] | None = None,
    ):
        layers_selected = (
            f"all layers" if select_layers is None else f"layers {select_layers}"
        )
        logging.info(
            f"quantizing {layers_selected} of model to {goal_dtype} (linear) ..."
        )
        quantize_layers(self.model, goal_dtype, self.device, select_layers)

    def bnb_quantize(self, num_bytes: int, nf4: bool = False):
        assert num_bytes in [4, 8], "only [4, 8] bytes supported"
        quant_type = "nf4" if nf4 else "linear"
        logging.info(
            f"quantizing model to {num_bytes} bytes using bits_and_bytes {quant_type} ..."
        )

        quant_type = "nf4" if nf4 else "fp4"
        if num_bytes == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type=quant_type
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_8bit_quant_type=quant_type
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.mod_path, quantization_config=bnb_config, torch_dtype=torch.float32
        )

    def memory_footprint(self):
        return self.model.get_memory_footprint()  # type: ignore

    def forward(self, sentences: str):
        encodings = self.tokenizer(sentences, return_tensors="pt", padding=True)
        input_ids = encodings["input_ids"].to(self.device)  # type: ignore
        attention_mask = encodings["attention_mask"].to(self.device)  # type: ignore

        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)  # type: ignore
        return outputs
