import logging
from enum import Enum
import argparse

import torch

from src.data import PennTreeBank, Wikipedia, get_dataloader
from src.model import AutoModel
from src.utils import calculate_perplexity


class QuantisationType(Enum):
    custom = "custom"
    custom_selective = "custom_selective"


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


def main(quantisation_type: QuantisationType):
    dataloader = get_dataloader(PennTreeBank(3000), batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")
    logging.info(f"device: {device}")
    model = AutoModel("./models", device, "gpt-neo")

    logging.info("running pre-trained model ... ")
    evaluate_model(model, dataloader, device)

    match quantisation_type:
        case QuantisationType.custom:
            select_layers = None
        case QuantisationType.custom_selective:
            # selecting self-attention layers for now
            select_layers = [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ]
        case _:
            raise ValueError("Invalid quantisation type")

    model.quantize_custom(torch.int8, select_layers=select_layers)

    logging.info("running quantized model ... ")
    evaluate_model(model, dataloader, device)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(asctime)s : %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="evaluate model with different quantization types"
    )
    parser.add_argument(
        "--q_type",
        type=lambda x: QuantisationType[x],
        choices=list(QuantisationType),
        default=QuantisationType.custom,
        help="type of quantization to apply",
        required=True,
    )
    args = parser.parse_args()

    main(args.q_type)
