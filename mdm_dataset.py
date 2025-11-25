import os
import torch
import numpy as np
import pydantic
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata

from dataclasses import dataclass

@dataclass
class MDMDatasetConfig:
    seed: int
    file_path: str
    tokenizer: any  # Can't type hint tokenizer easily
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int
    rank: int
    num_replicas: int
    max_size: int = -1
    input_output_separator: str = "###"
    use_puzzle_embeddings: bool = False


class MDMDataset(IterableDataset):
    def __init__(self, config: MDMDatasetConfig):
        super().__init__()
        self.config = config
        
        assert os.path.isfile(config.file_path), f"Input file path {config.file_path} not found"

        # Load and tokenize data
        with open(config.file_path, encoding="utf-8") as f:
            lines = [
                line.strip().split(config.input_output_separator) for line in f.readlines()
            ]
        if config.max_size > 0:
            print(f"truncated to {config.max_size}")
            lines = lines[:config.max_size]

        self.X = []
        self.y = []
        for X, y in lines:
            self.X.append(config.tokenizer([X], add_special_tokens=True)["input_ids"][0])
            self.y.append(config.tokenizer([y], add_special_tokens=True)["input_ids"][0])

        # Ensure all sequences have the same length (pad to max length)
        max_len = max(len(x) for x in self.X + self.y)
        pad_id = config.tokenizer.pad_token_id if config.tokenizer.pad_token_id is not None else 0
        
        # Pad sequences
        self.X = [x + [pad_id] * (max_len - len(x)) for x in self.X]
        self.y = [y + [pad_id] * (max_len - len(y)) for y in self.y]
        
        # Convert to numpy arrays
        self.X = np.array(self.X, dtype=np.int32)
        self.y = np.array(self.y, dtype=np.int32)

        # Create metadata
        set_name = "test" if config.test_set_mode else "train"
        num_puzzle_identifiers = 1 if config.use_puzzle_embeddings else 0
        self.metadata = PuzzleDatasetMetadata(
            pad_id=pad_id,
            ignore_label_id=None,  # disable ignore-label remapping
            blank_identifier_id=0,  # any id that won't be used (no puzzle IDs anyway)
            vocab_size=config.tokenizer.vocab_size,
            seq_len=max_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            total_groups=1,  # treat the whole dataset as one group
            mean_puzzle_examples=len(self.X),  # so total_groups * mean_puzzle_examples ~= dataset size
            total_puzzles=len(self.X),  # arbitrary but consistent
            sets=[set_name],
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, \
            f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State for iteration
        self._iters = 0

    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) if isinstance(v, np.ndarray) else v for k, v in batch.items()}

        # Convert ignore label IDs (not needed here since ignore_label_id is None)
        # if self.metadata.ignore_label_id is not None:
        #     batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,
                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _iter_test(self):
        total_examples = len(self.X)
        set_name = self.metadata.sets[0]

        # Load examples sequentially
        start_index = 0
        while start_index < total_examples:
            # Compute indices
            end_index = min(total_examples, start_index + self.config.global_batch_size)
            
            local_start = start_index + self.config.rank * self.local_batch_size
            local_end = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
            
            if local_start >= local_end:
                # This rank has no data for this batch
                start_index += self.config.global_batch_size
                continue
            
            # Create puzzle identifiers (all zeros since we don't have puzzle structure)
            puzzle_indices = np.zeros(local_end - local_start, dtype=np.int32)
            
            batch = self._collate_batch({
                "inputs": self.X[local_start:local_end],
                "labels": self.y[local_start:local_end],
                "puzzle_identifiers": puzzle_indices
            })

            yield set_name, batch, end_index - start_index
            
            # Advance to next batch
            start_index += self.config.global_batch_size

    def _iter_train(self):
        set_name = self.metadata.sets[0]
        
        # Increase epoch count
        self._iters += 1

        # Randomly shuffle indices
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
        
        # Create permutation for epochs_per_iter epochs
        indices = np.concatenate([
            rng.permutation(len(self.X)) for _ in range(self.config.epochs_per_iter)
        ])
        
        start_index = 0
        while start_index < len(indices):
            # Get batch indices
            batch_indices = indices[start_index:start_index + self.config.global_batch_size]
            
            # Drop last batch if incomplete
            if len(batch_indices) < self.config.global_batch_size:
                break
            
            # Select current rank's portion
            local_batch_indices = batch_indices[
                self.config.rank * self.local_batch_size: 
                (self.config.rank + 1) * self.local_batch_size
            ]
            
            # Create puzzle identifiers (all zeros since we don't have puzzle structure)
            puzzle_indices = np.zeros(len(local_batch_indices), dtype=np.int32)
            
            batch = self._collate_batch({
                "inputs": self.X[local_batch_indices],
                "labels": self.y[local_batch_indices],
                "puzzle_identifiers": puzzle_indices
            })

            yield set_name, batch, len(batch_indices)
            
            start_index += self.config.global_batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, \
            "Multithreaded data loading is not currently supported."
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()


if __name__ == "__main__":
    pass
