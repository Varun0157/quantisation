from typing import Tuple
import time
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import QuantisationType


def calculate_perplexity(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    model.eval()

    total_samples = len(dataloader.dataset)  # type: ignore
    total_loss = 0.0

    total_latency = 0.0

    for sentences in tqdm(dataloader, "calculating perplexity...", ascii=" ▖▘▝▗▚▞█"):
        with torch.no_grad():
            start_time = time.time()
            outputs = model(sentences)
            time_taken = time.time() - start_time
            total_latency += time_taken

            loss = outputs.loss

        assert not torch.isnan(loss).any(), f"NaN loss in batch {sentences}"
        total_loss += loss.item()

    perplexity = 2 ** (total_loss / total_samples)
    average_latency = total_latency / total_samples

    return perplexity, average_latency


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantisation script")
    parser.add_argument(
        "--q_type",
        type=QuantisationType,
        choices=list(QuantisationType),
        default=QuantisationType.none,
        help="Type of quantisation to apply",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU",
    )
    return parser


def get_logging_format() -> str:
    return "%(levelname)s - %(asctime)s : %(message)s"
