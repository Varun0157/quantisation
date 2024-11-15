import math
import time
from typing import Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_perplexity(
    model: nn.Module, dataloader: DataLoader
) -> Tuple[float, float]:
    model.eval()

    total_words = len(dataloader.dataset)  # type: ignore
    total_loss = 0.0
    total_inference_time = 0.0

    for sentences in tqdm(dataloader, "calculating perplexity...", ascii=" ▖▘▝▗▚▞█"):
        with torch.no_grad():
            start_time = time.time()
            outputs = model(sentences)
            total_inference_time += time.time() - start_time

            loss = outputs.loss

        assert not torch.isnan(loss).any(), f"NaN loss in batch {sentences}"
        total_loss += loss.item()

    assert total_words > 0, "no successful inferences"
    perplexity = math.exp(total_loss / total_words)

    average_inference_time = total_inference_time / total_words
    return perplexity, average_inference_time
