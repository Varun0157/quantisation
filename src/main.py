import logging

from numpy import average
import torch

from src.data import PennTreeBank, Wikipedia, get_dataloader
from src.model import AutoModel
from src.utils import calculate_perplexity


def evaluate_model(
    model: AutoModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
):
    memory = model.memory_footprint()
    perplexity, average_latency = calculate_perplexity(model, dataloader, device)
    logging.info(f"perplexity: {perplexity}")
    logging.info(f"average latency: {average_latency}")
    logging.info(f"memory footprint: {memory / 1e6} MB")


def main():
    dataloader = get_dataloader(PennTreeBank(3000), batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    model = AutoModel("./models", device, "gpt-neo")

    logging.info("running pre-trained model ... ")
    evaluate_model(model, dataloader, device)

    model.quantize_custom(torch.int8)

    logging.info("running quantized model ... ")
    evaluate_model(model, dataloader, device)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(asctime)s : %(message)s"
    )
    main()
