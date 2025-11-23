"""
Non-IID Data Partitioning Module

Implements three types of data heterogeneity:
1. Topic Skew - Each client gets data from specific domains
2. Style Skew - Each client gets data with different writing styles
3. Token Distribution Skew - Artificially manipulated token frequencies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import random
from abc import ABC, abstractmethod


class DataPartitioner(ABC):
    """Abstract base class for data partitioning strategies."""

    def __init__(self, num_clients: int, seed: int = 42):
        self.num_clients = num_clients
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    @abstractmethod
    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """
        Partition dataset indices among clients.

        Args:
            dataset: The dataset to partition

        Returns:
            Dictionary mapping client_id -> list of data indices
        """
        pass

    def get_statistics(self, dataset: Any, partition: Dict[int, List[int]]) -> Dict:
        """Get statistics about the partition."""
        stats = {
            'num_clients': len(partition),
            'samples_per_client': {cid: len(indices) for cid, indices in partition.items()},
            'total_samples': sum(len(indices) for indices in partition.values()),
        }
        return stats


class IIDPartitioner(DataPartitioner):
    """Randomly partition data uniformly across clients (IID baseline)."""

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        num_samples = len(dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        partition = {}
        samples_per_client = num_samples // self.num_clients

        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_id == self.num_clients - 1:
                end_idx = num_samples
            partition[client_id] = indices[start_idx:end_idx]

        return partition


class TopicSkewPartitioner(DataPartitioner):
    """
    Topic Skew: Each client receives data from specific topics/domains.

    This creates vocabulary and concept mismatch between clients.
    For AG News: sports, politics, tech, health categories.
    """

    def __init__(self, num_clients: int, label_column: str = 'label',
                 alpha: float = 0.1, seed: int = 42):
        """
        Args:
            num_clients: Number of FL clients
            label_column: Column name for topic/category labels
            alpha: Dirichlet concentration parameter (lower = more skewed)
            seed: Random seed
        """
        super().__init__(num_clients, seed)
        self.label_column = label_column
        self.alpha = alpha

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """Partition using Dirichlet distribution over labels."""
        # Get labels
        if hasattr(dataset, 'features') and self.label_column in dataset.features:
            labels = np.array(dataset[self.label_column])
        else:
            # Try to extract labels
            labels = np.array([sample.get(self.label_column, 0) for sample in dataset])

        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Create label -> indices mapping
        label_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_indices[label].append(idx)

        # Dirichlet distribution for label proportions per client
        partition = {i: [] for i in range(self.num_clients)}

        for label in range(num_classes):
            indices = np.array(label_indices[label])
            np.random.shuffle(indices)

            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(indices)).astype(int)

            # Adjust for rounding errors
            proportions[-1] = len(indices) - proportions[:-1].sum()

            # Assign to clients
            start = 0
            for client_id, num in enumerate(proportions):
                partition[client_id].extend(indices[start:start + num].tolist())
                start += num

        # Shuffle each client's data
        for client_id in partition:
            np.random.shuffle(partition[client_id])

        # Ensure minimum samples per client (redistribute from largest)
        min_samples = max(1, num_samples // (self.num_clients * 10))  # At least 10% of fair share
        for client_id in partition:
            while len(partition[client_id]) < min_samples:
                # Find client with most samples
                largest = max(partition.keys(), key=lambda k: len(partition[k]))
                if len(partition[largest]) > min_samples + 1:
                    # Move one sample
                    partition[client_id].append(partition[largest].pop())
                else:
                    break

        return partition

    def get_statistics(self, dataset: Any, partition: Dict[int, List[int]]) -> Dict:
        """Enhanced statistics with label distribution per client."""
        stats = super().get_statistics(dataset, partition)

        # Get labels
        if hasattr(dataset, 'features') and self.label_column in dataset.features:
            labels = np.array(dataset[self.label_column])
        else:
            labels = np.array([sample.get(self.label_column, 0) for sample in dataset])

        # Label distribution per client
        stats['label_distribution'] = {}
        for client_id, indices in partition.items():
            client_labels = labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            stats['label_distribution'][client_id] = dict(zip(unique.tolist(), counts.tolist()))

        return stats


class StyleSkewPartitioner(DataPartitioner):
    """
    Style Skew: Partition by writing style characteristics.

    Clients differ in:
    - Formal vs informal language
    - Sentence length
    - Vocabulary complexity
    """

    def __init__(self, num_clients: int, text_column: str = 'text',
                 style_criteria: str = 'length', seed: int = 42):
        """
        Args:
            num_clients: Number of FL clients
            text_column: Column name for text data
            style_criteria: 'length', 'formality', or 'complexity'
            seed: Random seed
        """
        super().__init__(num_clients, seed)
        self.text_column = text_column
        self.style_criteria = style_criteria

    def _compute_style_score(self, text: str) -> float:
        """Compute a style score for text."""
        if self.style_criteria == 'length':
            return len(text.split())
        elif self.style_criteria == 'formality':
            # Simple heuristic: ratio of formal punctuation and capitalization
            formal_chars = sum(1 for c in text if c in '.,:;')
            return formal_chars / max(len(text), 1)
        elif self.style_criteria == 'complexity':
            # Average word length as complexity proxy
            words = text.split()
            if not words:
                return 0
            return sum(len(w) for w in words) / len(words)
        return 0

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """Partition by style characteristics."""
        # Extract texts
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            texts = [dataset[i][self.text_column] if isinstance(dataset[i], dict)
                    else dataset[i] for i in range(len(dataset))]
        else:
            texts = list(dataset[self.text_column])

        # Compute style scores
        scores = [(idx, self._compute_style_score(text)) for idx, text in enumerate(texts)]
        scores.sort(key=lambda x: x[1])

        # Partition by score ranges
        partition = {i: [] for i in range(self.num_clients)}
        samples_per_client = len(scores) // self.num_clients

        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_id == self.num_clients - 1:
                end_idx = len(scores)

            partition[client_id] = [scores[i][0] for i in range(start_idx, end_idx)]

        return partition


class TokenDistributionSkewPartitioner(DataPartitioner):
    """
    Token Distribution Skew: Artificially manipulate token frequencies.

    Creates gradient bias by:
    - Filtering by token frequency
    - Heavy truncation variations
    - Stopword manipulation
    """

    def __init__(self, num_clients: int, text_column: str = 'text',
                 skew_type: str = 'frequency', seed: int = 42):
        """
        Args:
            num_clients: Number of FL clients
            text_column: Column name for text data
            skew_type: 'frequency', 'truncation', or 'vocabulary'
            seed: Random seed
        """
        super().__init__(num_clients, seed)
        self.text_column = text_column
        self.skew_type = skew_type

    def _get_token_stats(self, texts: List[str]) -> Dict[str, float]:
        """Get token frequency statistics."""
        from collections import Counter
        all_tokens = []
        for text in texts:
            all_tokens.extend(text.lower().split())

        counter = Counter(all_tokens)
        total = sum(counter.values())
        return {token: count / total for token, count in counter.items()}

    def _compute_text_frequency_score(self, text: str, token_freq: Dict[str, float]) -> float:
        """Score text by average token frequency."""
        tokens = text.lower().split()
        if not tokens:
            return 0
        scores = [token_freq.get(t, 0) for t in tokens]
        return sum(scores) / len(scores)

    def partition(self, dataset: Any) -> Dict[int, List[int]]:
        """Partition by token distribution characteristics."""
        # Extract texts
        if hasattr(dataset, '__getitem__') and hasattr(dataset, '__len__'):
            texts = [dataset[i][self.text_column] if isinstance(dataset[i], dict)
                    else dataset[i] for i in range(len(dataset))]
        else:
            texts = list(dataset[self.text_column])

        if self.skew_type == 'frequency':
            # Partition by token frequency score
            token_freq = self._get_token_stats(texts)
            scores = [(idx, self._compute_text_frequency_score(text, token_freq))
                     for idx, text in enumerate(texts)]
        elif self.skew_type == 'truncation':
            # Partition by text length (truncation proxy)
            scores = [(idx, len(text)) for idx, text in enumerate(texts)]
        else:  # vocabulary
            # Partition by unique token ratio
            scores = [(idx, len(set(text.lower().split())) / max(len(text.split()), 1))
                     for idx, text in enumerate(texts)]

        scores.sort(key=lambda x: x[1])

        # Partition by score ranges
        partition = {i: [] for i in range(self.num_clients)}
        samples_per_client = len(scores) // self.num_clients

        for client_id in range(self.num_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client
            if client_id == self.num_clients - 1:
                end_idx = len(scores)

            partition[client_id] = [scores[i][0] for i in range(start_idx, end_idx)]

        return partition


class NonIIDPartitioner:
    """
    Factory class for creating various non-IID partitioning strategies.
    """

    STRATEGIES = {
        'iid': IIDPartitioner,
        'topic_skew': TopicSkewPartitioner,
        'style_skew': StyleSkewPartitioner,
        'token_skew': TokenDistributionSkewPartitioner,
    }

    @classmethod
    def create(cls, strategy: str, num_clients: int, **kwargs) -> DataPartitioner:
        """
        Create a partitioner with the specified strategy.

        Args:
            strategy: One of 'iid', 'topic_skew', 'style_skew', 'token_skew'
            num_clients: Number of FL clients
            **kwargs: Additional arguments for the specific partitioner

        Returns:
            DataPartitioner instance
        """
        if strategy not in cls.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. "
                           f"Available: {list(cls.STRATEGIES.keys())}")

        return cls.STRATEGIES[strategy](num_clients=num_clients, **kwargs)

    @classmethod
    def available_strategies(cls) -> List[str]:
        """Return list of available partitioning strategies."""
        return list(cls.STRATEGIES.keys())
