import logging

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset


class PennTreeBank(Dataset):
    def __init__(self) -> None:
        logging.info("loading Penn Tree Bank dataset...")
        dataset = load_dataset("ptb_text_only", "penn_treebank")
        assert type(dataset) == DatasetDict

        self.texts = (
            dataset["test"]["sentence"]
            + dataset["validation"]["sentence"]
            + dataset["train"]["sentence"]
        )
        self.texts = self.texts[:1000]
        logging.info("Penn Tree Bank dataset loaded")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> str:
        return self.texts[idx]


def get_dataloader(dataset: Dataset, batch_size: int = 1) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
