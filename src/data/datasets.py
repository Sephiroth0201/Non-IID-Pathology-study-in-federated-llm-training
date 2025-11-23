"""
Dataset Module for AG News Classification

AG News has 4 classes: World, Sports, Business, Sci/Tech
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from typing import Dict, List, Tuple, Any
import numpy as np


class AGNewsDataset(Dataset):
    """AG News dataset for classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_ag_news(max_samples: int = 0, split: str = 'train') -> Tuple[List[str], List[int]]:
    """Load AG News dataset."""
    dataset = load_dataset('ag_news', split=split)
    texts = list(dataset['text'])
    labels = list(dataset['label'])

    if max_samples > 0:
        texts = texts[:max_samples]
        labels = labels[:max_samples]

    return texts, labels


class FederatedDataset:
    """Manages federated data distribution."""

    def __init__(self, dataset: Dataset, client_indices: Dict[int, List[int]], batch_size: int = 32):
        self.dataset = dataset
        self.client_indices = client_indices
        self.batch_size = batch_size

    def get_client_loader(self, client_id: int, shuffle: bool = True) -> DataLoader:
        """Get dataloader for a specific client."""
        indices = self.client_indices[client_id]
        subset = Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=shuffle)

    def get_client_size(self, client_id: int) -> int:
        return len(self.client_indices[client_id])

    def get_all_clients(self) -> List[int]:
        return list(self.client_indices.keys())


def create_federated_dataset(
    tokenizer,
    partition: Dict[int, List[int]],
    max_samples: int = 2000,
    max_length: int = 64,
    batch_size: int = 32
) -> Tuple[FederatedDataset, DataLoader]:
    """
    Create federated dataset and test loader.

    Returns:
        (federated_train_data, test_loader)
    """
    # Load train data
    texts, labels = load_ag_news(max_samples, 'train')
    train_dataset = AGNewsDataset(texts, labels, tokenizer, max_length)

    # Create federated dataset
    fed_data = FederatedDataset(train_dataset, partition, batch_size)

    # Load test data (small subset)
    test_texts, test_labels = load_ag_news(500, 'test')
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return fed_data, test_loader


# Simple dataset wrapper for partitioning
class SimpleDataset:
    def __init__(self, data: Dict[str, List]):
        self.data = data

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        return {k: v[key] for k, v in self.data.items()}

    @property
    def features(self):
        return self.data.keys()
