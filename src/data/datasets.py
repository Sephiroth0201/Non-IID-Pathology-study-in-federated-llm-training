"""
Dataset loading and preprocessing for federated LLM training.

Supports:
- WikiText-2 (language modeling)
- AG News (text classification with topic labels)
- Custom text datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np


class TextDataset(Dataset):
    """Custom dataset for tokenized text data."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
        labels: Optional[List[int]] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        # For causal LM, labels are the same as input_ids
        item['labels'] = item['input_ids'].clone()

        if self.labels is not None:
            item['class_label'] = torch.tensor(self.labels[idx])

        return item


class FederatedDataset:
    """
    Wrapper for federated dataset management.
    Handles client-specific data subsets.
    """

    def __init__(
        self,
        dataset: Dataset,
        client_indices: Dict[int, List[int]],
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
        max_length: int = 128
    ):
        self.dataset = dataset
        self.client_indices = client_indices
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_clients = len(client_indices)

    def get_client_dataloader(
        self,
        client_id: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Get DataLoader for a specific client."""
        if client_id not in self.client_indices:
            raise ValueError(f"Client {client_id} not found")

        indices = self.client_indices[client_id]
        client_subset = Subset(self.dataset, indices)

        # Disable pin_memory for MPS (Apple Silicon)
        import torch
        use_pin_memory = torch.cuda.is_available()

        return DataLoader(
            client_subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_pin_memory
        )

    def get_client_data_size(self, client_id: int) -> int:
        """Get number of samples for a client."""
        return len(self.client_indices[client_id])

    def get_all_client_ids(self) -> List[int]:
        """Get list of all client IDs."""
        return list(self.client_indices.keys())


def load_wikitext(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    split: str = 'train'
) -> Tuple[Dataset, List[str]]:
    """
    Load WikiText-2 dataset.

    Returns:
        Tuple of (tokenized dataset, raw texts)
    """
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)

    # Filter out empty lines
    texts = [text for text in dataset['text'] if text.strip()]

    # Create tokenized dataset
    tokenized_dataset = TextDataset(texts, tokenizer, max_length)

    return tokenized_dataset, texts


def load_ag_news(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    split: str = 'train'
) -> Tuple[Dataset, List[str], List[int]]:
    """
    Load AG News dataset with topic labels.

    Categories: World (0), Sports (1), Business (2), Sci/Tech (3)

    Returns:
        Tuple of (tokenized dataset, raw texts, labels)
    """
    dataset = load_dataset('ag_news', split=split)

    texts = dataset['text']
    labels = dataset['label']

    tokenized_dataset = TextDataset(texts, tokenizer, max_length, labels)

    return tokenized_dataset, texts, labels


def load_federated_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    partition: Dict[int, List[int]],
    batch_size: int = 8,
    max_length: int = 128,
    split: str = 'train'
) -> FederatedDataset:
    """
    Load a dataset and create federated dataset with given partition.

    Args:
        dataset_name: 'wikitext' or 'ag_news'
        tokenizer: Tokenizer for text processing
        partition: Client -> indices mapping from partitioner
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        split: Dataset split to use

    Returns:
        FederatedDataset instance
    """
    if dataset_name == 'wikitext':
        dataset, _ = load_wikitext(tokenizer, max_length, split)
    elif dataset_name == 'ag_news':
        dataset, _, _ = load_ag_news(tokenizer, max_length, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return FederatedDataset(
        dataset=dataset,
        client_indices=partition,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )


def create_raw_dataset_for_partitioning(
    dataset_name: str,
    split: str = 'train',
    max_samples: int = 0
) -> Any:
    """
    Load raw dataset (without tokenization) for partitioning.

    This is used before tokenization to partition by text characteristics.

    Args:
        dataset_name: Name of dataset
        split: Dataset split
        max_samples: Maximum samples to use (0 = all)
    """
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        # Filter empty lines and create indexable structure
        texts = [text for text in dataset['text'] if text.strip()]
        if max_samples > 0:
            texts = texts[:max_samples]
        return {'text': texts, 'label': [0] * len(texts)}
    elif dataset_name == 'ag_news':
        dataset = load_dataset('ag_news', split=split)
        texts = list(dataset['text'])
        labels = list(dataset['label'])
        if max_samples > 0:
            texts = texts[:max_samples]
            labels = labels[:max_samples]
        return {'text': texts, 'label': labels}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class SimpleDataset:
    """Simple wrapper to make dict-like data indexable."""

    def __init__(self, data: Dict[str, List]):
        self.data = data
        self._length = len(list(data.values())[0])

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, key):
        # Support both integer indexing and column name access
        if isinstance(key, str):
            # Column access: dataset['label'] returns list of all labels
            return self.data[key]
        else:
            # Integer indexing: dataset[0] returns dict for that sample
            return {k: v[key] for k, v in self.data.items()}

    @property
    def features(self):
        return self.data.keys()


def prepare_federated_data(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    partitioner: Any,
    batch_size: int = 8,
    max_length: int = 128,
    train_split: str = 'train',
    test_split: str = 'test'
) -> Tuple[FederatedDataset, DataLoader]:
    """
    Complete pipeline to prepare federated training and test data.

    Args:
        dataset_name: Dataset to load
        tokenizer: Tokenizer for text processing
        partitioner: DataPartitioner instance
        batch_size: Batch size
        max_length: Maximum sequence length
        train_split: Training split name
        test_split: Test split name

    Returns:
        Tuple of (federated_train_dataset, test_dataloader)
    """
    # Load raw data for partitioning
    raw_data = create_raw_dataset_for_partitioning(dataset_name, train_split)
    raw_dataset = SimpleDataset(raw_data)

    # Partition the data
    partition = partitioner.partition(raw_dataset)

    # Create federated dataset
    federated_dataset = load_federated_dataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        partition=partition,
        batch_size=batch_size,
        max_length=max_length,
        split=train_split
    )

    # Load test data
    if dataset_name == 'wikitext':
        test_dataset, _ = load_wikitext(tokenizer, max_length, test_split)
    else:
        test_dataset, _, _ = load_ag_news(tokenizer, max_length, test_split)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return federated_dataset, test_loader
