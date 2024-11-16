import logging

import torch

from src.data import PennTreeBank, get_dataloader
from src.utils import calculate_perplexity, get_parser, get_logging_format
from src.model import AutoModel, QuantisationType
from src.quantize import get_model


def get_method_name(quant_method: QuantisationType):
    return f"gpt-neo_{quant_method.value}"


def evaluate_model(
    model: AutoModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    quant_method: QuantisationType,
):
    memory = model.memory_footprint()
    perplexity, average_latency = calculate_perplexity(model, dataloader, device)
    logging.info(f"perplexity: {perplexity}")
    logging.info(f"average latency: {average_latency}")
    logging.info(f"memory footprint: {memory / 1e6} MB")

    res_file_name = f"./results/{get_method_name(quant_method)}.md"
    with open(res_file_name, "w") as f:
        f.write(f"perplexity: {perplexity}\n")
        f.write(f"average latency: {average_latency}\n")
        f.write(f"memory_footprint: {memory / 1e6} MB")


def main(quantisation_type: QuantisationType, cpu: bool = False):
    device = torch.device("cuda" if not cpu and torch.cuda.is_available() else "cpu")
    
    dataloader = get_dataloader(PennTreeBank(3000), batch_size=1)
    
    model = get_model(quantisation_type, cpu)
    model.to(device)
    model.load_state_dict(
        torch.load(
            f"./quantized/{get_method_name(quantisation_type)}.pt", weights_only=True
        ),
        strict=False,
    )
    logging.info("model loaded")
    
    evaluate_model(model, dataloader, device, quantisation_type)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=get_logging_format())
    parser = get_parser()
    args = parser.parse_args()

    main(args.q_type, args.cpu)
