import logging

import torch

from src.data import PennTreeBank, get_dataloader
from src.model import GPTNeo
from src.utils import calculate_perplexity


def run_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    perplexity, avg_time = calculate_perplexity(model, dataloader)
    print(f"perplexity: {perplexity}")
    print(f"average inference time: {avg_time}")

    mem_used_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"memory used: {mem_used_mb:.2f} MB")


def main():
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTNeo("./models", device)

    dataloader = get_dataloader(PennTreeBank(), batch_size=1)
    run_model(model, dataloader)

    model.quantize()

    run_model(model, dataloader)


if __name__ == "__main__":
    main()
