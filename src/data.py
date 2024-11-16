import logging
from typing import List

from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader, Dataset


def clean(texts: List[str]) -> List[str]:
    cleaned = []
    for text in texts:
        num_words = len(text.split())
        if num_words < 2:
            continue
        cleaned.append(text)

    return cleaned


# todo: this looks easily generalizable to other datasets, look at the init. Could be a good refactoring exercise.


class PennTreeBank(Dataset):
    name = "PennTreeBank"

    def __init__(self, num_sentences: int | None = None) -> None:
        logging.info(f"Loading {self.name} dataset...")
        dataset = load_dataset("ptb_text_only", "penn_treebank")
        assert type(dataset) == DatasetDict

        self.texts = clean(
            dataset["test"]["sentence"]
            + dataset["validation"]["sentence"]
            + dataset["train"]["sentence"]
        )

        assert num_sentences is None or num_sentences <= len(
            self.texts
        ), f"only {len(self.texts)} sentences present"

        if num_sentences is not None:
            self.texts = self.texts[:num_sentences]  # todo: random sample

        logging.info(f"{self.name} dataset loaded")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> str:
        return self.texts[idx]


class Wikipedia(Dataset):
    name = "Wikipedia"

    def __init__(self, num_sentences: int | None = None) -> None:
        logging.info(f"Loading {self.name} dataset...")
        dataset = load_dataset("wikipedia", "20200501.en")
        assert type(dataset) == DatasetDict

        self.texts = clean(
            dataset["train"]["text"]
            + dataset["validation"]["text"]
            + dataset["test"]["text"]
        )

        assert num_sentences is None or num_sentences <= len(
            self.texts
        ), f"only {len(self.texts)} sentences present"

        if num_sentences is not None:
            self.texts = self.texts[:num_sentences]  # todo: random sample

        logging.info(f"{self.name} dataset loaded")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx) -> str:
        return self.texts[idx]


def get_dataloader(dataset: Dataset, batch_size: int = 1) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
