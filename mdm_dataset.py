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
    use_commutative_augmentation: bool = False  # Enable commutative augmentation (swap a*b to b*a)
    # Cyclic data mixing: optional additional dataset that gets sampled and mixed
    additional_dataset_path: str = ""  # Path to additional dataset file (empty = disabled)
    cyclic_subset_size: int = 0  # Number of samples to sample from additional dataset (0 = disabled)
    cyclic_refresh_epochs: int = 10  # Number of epochs before refreshing the subset


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
        # Store original strings only if augmentation is enabled (to save memory)
        self.original_X = [] if config.use_commutative_augmentation else None
        for X, y in lines:
            if config.use_commutative_augmentation:
                self.original_X.append(X)  # Store original string for augmentation
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
        
        # Cyclic data mixing: load additional dataset if provided
        self.additional_X = None
        self.additional_y = None
        self.additional_original_X = None
        self.current_subset_indices = None
        self.epochs_since_refresh = 0
        
        if (not config.test_set_mode and 
            config.additional_dataset_path and 
            config.cyclic_subset_size > 0 and 
            os.path.isfile(config.additional_dataset_path)):
            
            print(f"Loading additional dataset from {config.additional_dataset_path}")
            with open(config.additional_dataset_path, encoding="utf-8") as f:
                additional_lines = [
                    line.strip().split(config.input_output_separator) for line in f.readlines()
                ]
            
            # Tokenize additional dataset
            additional_X_list = []
            additional_y_list = []
            additional_original_X_list = [] if config.use_commutative_augmentation else None
            
            for X, y in additional_lines:
                if config.use_commutative_augmentation:
                    additional_original_X_list.append(X)
                additional_X_list.append(config.tokenizer([X], add_special_tokens=True)["input_ids"][0])
                additional_y_list.append(config.tokenizer([y], add_special_tokens=True)["input_ids"][0])
            
            # Pad to same length as main dataset
            additional_X_list = [x + [pad_id] * (max_len - len(x)) for x in additional_X_list]
            additional_y_list = [y + [pad_id] * (max_len - len(y)) for y in additional_y_list]
            
            # Convert to numpy arrays
            self.additional_X = np.array(additional_X_list, dtype=np.int32)
            self.additional_y = np.array(additional_y_list, dtype=np.int32)
            if config.use_commutative_augmentation:
                self.additional_original_X = additional_original_X_list
            
            print(f"Loaded {len(self.additional_X)} samples from additional dataset")
            print(f"Will sample {min(config.cyclic_subset_size, len(self.additional_X))} samples every {config.cyclic_refresh_epochs} epochs")

    def _swap_operands(self, input_str: str) -> str:
        """
        Swaps operands in multiplication input string.
        Input format: " {a_str} * {b_str}### {res_str}"
        Output format: " {b_str} * {a_str}### {res_str}"
        """
        if " * " not in input_str or self.config.input_output_separator not in input_str:
            return input_str  # Return as-is if format is unexpected
        
        parts = input_str.split(self.config.input_output_separator, 1)
        if len(parts) != 2:
            return input_str
        
        input_part, output_part = parts
        input_part = input_part.strip()
        
        # Split on " * " to get operands
        if " * " not in input_part:
            return input_str
        
        operands = input_part.split(" * ", 1)
        if len(operands) != 2:
            return input_str
        
        a_str, b_str = operands
        
        # Swap operands
        swapped_input = f" {b_str} * {a_str}"
        return f"{swapped_input}{self.config.input_output_separator} {output_part.strip()}"

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
        
        # Check if we need to refresh the cyclic subset
        if (self.additional_X is not None and 
            self.config.cyclic_subset_size > 0):
            
            # Calculate total epochs processed (each iteration processes epochs_per_iter epochs)
            total_epochs_processed = self._iters * self.config.epochs_per_iter
            
            # Refresh subset every N epochs
            if (self.current_subset_indices is None or 
                (total_epochs_processed - self.epochs_since_refresh) >= self.config.cyclic_refresh_epochs):
                
                REFRESH_RNG_SEED_OFFSET = 1000000  
                rng_refresh = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters + REFRESH_RNG_SEED_OFFSET))
                subset_size = min(self.config.cyclic_subset_size, len(self.additional_X))
                self.current_subset_indices = rng_refresh.choice(
                    len(self.additional_X), 
                    size=subset_size, 
                    replace=False
                )
                self.epochs_since_refresh = total_epochs_processed
                if self.config.rank == 0:  # Only print from rank 0
                    print(f"[Iteration {self._iters}, Total Epochs {total_epochs_processed}] Refreshed cyclic subset: sampled {subset_size} samples from additional dataset")

        # Randomly shuffle indices
        rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
        
        # Create permutation for main dataset
        main_indices = np.concatenate([
            rng.permutation(len(self.X)) for _ in range(self.config.epochs_per_iter)
        ])
        
        # Mix with cyclic subset if available
        if self.current_subset_indices is not None:
            # Create indices for subset (offset by len(self.X) to distinguish from main)
            subset_indices = np.concatenate([
                self.current_subset_indices + len(self.X) for _ in range(self.config.epochs_per_iter)
            ])
            # Concatenate and shuffle together
            indices = np.concatenate([main_indices, subset_indices])
            indices = rng.permutation(indices)
        else:
            indices = main_indices
        
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
            
            # Process each index in order (maintain batch order)
            # Indices >= len(self.X) are from additional dataset
            batch_inputs_list = []
            batch_labels_list = []
            
            for idx in local_batch_indices:
                if idx < len(self.X):
                    # From main dataset
                    if self.config.use_commutative_augmentation:
                        original_str = self.original_X[idx]
                        # Randomly swap operands with 50% probability
                        if rng.random() < 0.5:
                            augmented_str = self._swap_operands(original_str)
                        else:
                            augmented_str = original_str
                        
                        # Re-tokenize the (possibly augmented) string
                        tokenized = self.config.tokenizer([augmented_str], add_special_tokens=True)["input_ids"][0]
                        # Pad to match max_len
                        pad_id = self.metadata.pad_id
                        if len(tokenized) < self.metadata.seq_len:
                            tokenized = tokenized + [pad_id] * (self.metadata.seq_len - len(tokenized))
                        elif len(tokenized) > self.metadata.seq_len:
                            tokenized = tokenized[:self.metadata.seq_len]
                        
                        batch_inputs_list.append(tokenized)
                        batch_labels_list.append(self.y[idx])
                    else:
                        batch_inputs_list.append(self.X[idx])
                        batch_labels_list.append(self.y[idx])
                else:
                    # From additional dataset (subtract offset)
                    additional_idx = idx - len(self.X)
                    if self.config.use_commutative_augmentation and self.additional_original_X is not None:
                        original_str = self.additional_original_X[additional_idx]
                        # Randomly swap operands with 50% probability
                        if rng.random() < 0.5:
                            augmented_str = self._swap_operands(original_str)
                        else:
                            augmented_str = original_str
                        
                        # Re-tokenize the (possibly augmented) string
                        tokenized = self.config.tokenizer([augmented_str], add_special_tokens=True)["input_ids"][0]
                        # Pad to match max_len
                        pad_id = self.metadata.pad_id
                        if len(tokenized) < self.metadata.seq_len:
                            tokenized = tokenized + [pad_id] * (self.metadata.seq_len - len(tokenized))
                        elif len(tokenized) > self.metadata.seq_len:
                            tokenized = tokenized[:self.metadata.seq_len]
                        
                        batch_inputs_list.append(tokenized)
                        batch_labels_list.append(self.additional_y[additional_idx])
                    else:
                        batch_inputs_list.append(self.additional_X[additional_idx])
                        batch_labels_list.append(self.additional_y[additional_idx])
            
            # Convert to arrays
            batch_inputs = np.array(batch_inputs_list, dtype=np.int32)
            batch_labels = np.array(batch_labels_list, dtype=np.int32)
            
            # Create puzzle identifiers (all zeros since we don't have puzzle structure)
            puzzle_indices = np.zeros(len(local_batch_indices), dtype=np.int32)
            
            batch = self._collate_batch({
                "inputs": batch_inputs,
                "labels": batch_labels,
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
