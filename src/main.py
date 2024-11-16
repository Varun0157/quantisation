import logging
from enum import Enum
import argparse

import torch

from src.data import PennTreeBank, Wikipedia, get_dataloader
from src.model import AutoModel
from src.utils import calculate_perplexity


class QuantisationType(Enum):
    none = "none"
    custom_whole = "custom_whole"
    custom_selective = "custom_selective"
    bnb_4 = "bnb_4"
    bnb_8 = "bnb_8"
    bnb_4_nf4 = "bnb_4_nf4"
    bnb_8_nf4 = "bnb_8_nf4"


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


def main(quantisation_type: QuantisationType, cpu: bool = False):
    dataloader = get_dataloader(PennTreeBank(3000), batch_size=1)

    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")
    model = AutoModel("./models", device, "gpt-neo")

    match quantisation_type:
        case QuantisationType.custom_whole:
            model.quantize_custom(torch.int8)
        case QuantisationType.custom_selective:
            # selecting self-attention layers for now
            select_layers = [
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
            ]
            model.quantize_custom(torch.int8, select_layers=select_layers)
        case QuantisationType.bnb_4:
            model.bnb_quantize(4)
        case QuantisationType.bnb_8:
            model.bnb_quantize(8)
        case QuantisationType.bnb_4_nf4:
            model.bnb_quantize(4, nf4=True)
        case QuantisationType.bnb_8_nf4:
            model.bnb_quantize(8, nf4=True)
        case QuantisationType.none:
            pass
        case _:
            raise ValueError("Invalid quantisation type")

    logging.info("running model ... ")
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
        default=QuantisationType.custom_whole,
        help="type of quantization to apply",
        required=True,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="use cpu instead of gpu",
    )
    args = parser.parse_args()

    main(args.q_type, args.cpu)
