"""
Federated Learning Clients: FedAvg and FedProx

FedAvg: Standard federated averaging (McMahan et al., 2017)
FedProx: Adds proximal term to handle heterogeneity (Li et al., 2020)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, Any
from tqdm import tqdm

from ..models.lora_model import get_model_params, set_model_params


class FederatedClient:
    """
    FedAvg Client - Standard federated learning.

    Each client trains locally on their data and returns updated parameters.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 2e-5,
        local_epochs: int = 2
    ):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs

    def train(self, global_params: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Train locally and return updated params."""
        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            pbar = tqdm(self.dataloader, desc=f"Client {self.client_id}", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                preds = outputs.logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

                pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        stats = {
            'client_id': self.client_id,
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'samples': total_samples
        }

        return get_model_params(self.model), stats


class FedProxClient(FederatedClient):
    """
    FedProx Client - Adds proximal term to handle data heterogeneity.

    Loss = CrossEntropy + (mu/2) * ||w - w_global||^2

    The proximal term keeps local updates close to global model,
    reducing client drift under non-IID data.
    """

    def __init__(self, client_id, model, dataloader, device, lr=2e-5, local_epochs=2, mu=0.1):
        super().__init__(client_id, model, dataloader, device, lr, local_epochs)
        self.mu = mu

    def train(self, global_params: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.local_epochs):
            pbar = tqdm(self.dataloader, desc=f"Client {self.client_id} (Prox)", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Proximal term: (mu/2) * ||w - w_global||^2
                prox_term = 0
                for name, param in self.model.named_parameters():
                    if name in global_params and param.requires_grad:
                        prox_term += ((param - global_params[name].to(self.device)) ** 2).sum()
                loss = loss + (self.mu / 2) * prox_term

                loss.backward()
                optimizer.step()

                preds = outputs.logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_loss += outputs.loss.item() * len(labels)  # Track CE loss only
                total_samples += len(labels)

                pbar.set_postfix({'loss': f'{outputs.loss.item():.3f}'})

        stats = {
            'client_id': self.client_id,
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'samples': total_samples
        }

        return get_model_params(self.model), stats
