"""
Model Module - DistilBERT for Text Classification

Lightweight model for fast federated learning experiments.
"""

import torch
import torch.nn as nn
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from typing import Dict, Optional, Tuple


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_model_and_tokenizer(
    num_labels: int = 4
) -> Tuple[nn.Module, DistilBertTokenizer]:
    """
    Load DistilBERT for sequence classification.

    Args:
        num_labels: Number of classification labels (4 for AG News)

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )

    # Freeze base layers, only train classifier (for speed)
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False

    # Keep last 2 transformer layers trainable
    for i, layer in enumerate(model.distilbert.transformer.layer):
        if i < 4:  # Freeze first 4 layers (out of 6)
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model, tokenizer


def get_model_params(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract trainable parameters."""
    return {name: param.data.clone() for name, param in model.named_parameters()
            if param.requires_grad}


def set_model_params(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Set trainable parameters."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict and param.requires_grad:
                param.copy_(state_dict[name])


def average_params(
    param_list: list,
    weights: Optional[list] = None
) -> Dict[str, torch.Tensor]:
    """Average parameters from multiple clients."""
    if weights is None:
        weights = [1.0 / len(param_list)] * len(param_list)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    averaged = {}
    for key in param_list[0]:
        averaged[key] = sum(w * p[key].float() for w, p in zip(weights, param_list))
    return averaged
