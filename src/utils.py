import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_perplexity(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()

    total_loss = 0.0
    total_words = len(dataloader.dataset)  # type: ignore

    for batch in dataloader:
        input_ids, attention_masks = batch
        input_ids = input_ids.to(model.device)
        attention_masks = attention_masks.to(model.device)

        labels = input_ids.clone()

        with torch.no_grad():
            outputs = model(input_ids, attention_masks, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    perplexity = math.exp(total_loss / total_words)
    return perplexity
