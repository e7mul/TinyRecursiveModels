import os
import torch
from torch.utils.data import Dataset

from dataset.common import PuzzleDatasetMetadata

class MDMDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        file_path,
        max_size=-1,
        input_output_separator: str = "###"
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = [
                line.strip().split(input_output_separator) for line in f.readlines()
            ]
        if max_size > 0:
            print(f"truncated to {max_size}")
            lines = lines[:max_size]

        self.X = []
        self.y = []
        for X, y in lines:
            self.X.append(tokenizer([X], add_special_tokens=True)["input_ids"][0])
            self.y.append(tokenizer([y], add_special_tokens=True)["input_ids"][0])

        self.metadata = PuzzleDatasetMetadata(
            seq_len=len(self.X[0]),
            vocab_size=tokenizer.vocab_size,
            pad_id=None,
            ignore_label_id=None,
            blank_identifier_id=None,
            num_puzzle_identifiers=None,
            total_groups=None,
            mean_puzzle_examples=None,
            total_puzzles=None,
            sets=None,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        input_ids = self.X[i]
        labels = self.y[i]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


if __name__ == "__main__":
    pass
