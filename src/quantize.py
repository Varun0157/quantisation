import logging
import os

import torch

from evaluate import get_method_name
from src.model import AutoModel, QuantisationType
from src.utils import get_logging_format, get_parser


def get_model(
    model_name: str, quantisation_type: QuantisationType, cpu: bool = False
) -> AutoModel:
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    model = AutoModel(model_name, device)

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
            model.bnb_quantize_linear(4)
        case QuantisationType.bnb_8:
            model.bnb_quantize_linear(8)
        case QuantisationType.bnb_4_nf4:
            model.bnb_quantize_nf4()
        case QuantisationType.none:
            pass
        case _:
            raise ValueError("Invalid quantisation type")

    return model


def main(quantisation_type: QuantisationType, cpu: bool = False):
    model_name = "EleutherAI/gpt-neo-125m"
    # model_name = "facebook/opt-125m"

    model = get_model(model_name, quantisation_type, cpu)
    logging.info(f"model: {model}")
    # save the model locally
    model_path = os.path.join("quantized", f"{get_method_name(quantisation_type)}.pt")
    logging.info("saving model ...")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=get_logging_format())
    parser = get_parser()
    args = parser.parse_args()

    main(args.q_type, args.cpu)
