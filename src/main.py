import logging 

import torch

from src.data import PennTreeBank, get_dataloader
from src.model import GPTNeo
from src.utils import calculate_perplexity


def main():
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")
    model = GPTNeo("./models", device)
    # model.quantize()

    dataset = PennTreeBank()
    dataloader = get_dataloader(dataset, batch_size=1)

    perplexity = calculate_perplexity(model, dataloader)
    print(f"perplexity: {perplexity}")


if __name__ == "__main__":
    main()
