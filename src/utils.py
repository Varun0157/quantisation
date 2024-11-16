from typing import Tuple

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_perplexity(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()

    total_samples = len(dataloader.dataset)  # type: ignore
    total_loss = 0.0

    total_latency = 0.0
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    for sentences in tqdm(dataloader, "calculating perplexity...", ascii=" ▖▘▝▗▚▞█"):
        with torch.no_grad():
            start_time.record(stream=torch.cuda.current_stream())
            outputs = model(sentences)
            end_time.record(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()

            loss = outputs.loss
            total_latency += start_time.elapsed_time(end_time)

        assert not torch.isnan(loss).any(), f"NaN loss in batch {sentences}"
        total_loss += loss.item()

    perplexity = 2 ** (total_loss / total_samples)
    average_latency = total_latency / total_samples

    return perplexity, average_latency
