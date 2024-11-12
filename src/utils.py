import math
import logging

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_perplexity(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()

    total_words = len(dataloader.dataset)  # type: ignore
    num_skipped_batches = 0
    total_loss = 0.0

    for batch in tqdm(dataloader, "calculating perplexity...", ascii=" ▖▘▝▗▚▞█"):
        sentences = batch

        with torch.no_grad():
            outputs = model(sentences)
            loss = outputs.loss

        if torch.isnan(loss).any():
            num_skipped_batches += 1
            total_words -= len(batch)
            continue

        total_loss += loss.item()

    if num_skipped_batches > 0:
        logging.warning(f"skipped {num_skipped_batches} batches with NaN loss")

    assert total_words > 0, "no successful inferences"
    perplexity = math.exp(total_loss / total_words)
    return perplexity
