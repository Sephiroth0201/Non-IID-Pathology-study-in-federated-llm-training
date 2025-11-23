"""
Federated Learning Client for Classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, Tuple, Any, Optional
from tqdm import tqdm

from ..models.lora_model import get_model_params, set_model_params


class FederatedClient:
    """Client for federated classification."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        lr: float = 2e-5,
        local_epochs: int = 1
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

                # Track metrics
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
    """FedProx client with proximal term."""

    def __init__(self, client_id, model, dataloader, device, lr=2e-5, local_epochs=1, mu=0.01):
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

                # Add proximal term
                prox = 0
                for name, param in self.model.named_parameters():
                    if name in global_params and param.requires_grad:
                        prox += ((param - global_params[name].to(self.device)) ** 2).sum()
                loss = loss + (self.mu / 2) * prox

                loss.backward()
                optimizer.step()

                preds = outputs.logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_loss += outputs.loss.item() * len(labels)
                total_samples += len(labels)

                pbar.set_postfix({'loss': f'{outputs.loss.item():.3f}'})

        stats = {
            'client_id': self.client_id,
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'samples': total_samples
        }

        return get_model_params(self.model), stats


class SCAFFOLDClient(FederatedClient):
    """SCAFFOLD client with control variates."""

    def __init__(self, client_id, model, dataloader, device, lr=2e-5, local_epochs=1):
        super().__init__(client_id, model, dataloader, device, lr, local_epochs)
        self.c_local = None  # Client control variate

    def init_control(self, params: Dict[str, torch.Tensor]):
        """Initialize control variate."""
        self.c_local = {k: torch.zeros_like(v) for k, v in params.items()}

    def train(
        self,
        global_params: Dict[str, torch.Tensor],
        c_global: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:

        if self.c_local is None:
            self.init_control(global_params)

        set_model_params(self.model, global_params)
        self.model.to(self.device)
        self.model.train()

        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

        total_loss = 0
        total_correct = 0
        total_samples = 0
        num_steps = 0

        for epoch in range(self.local_epochs):
            pbar = tqdm(self.dataloader, desc=f"Client {self.client_id} (SCAFFOLD)", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                # SCAFFOLD correction
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in c_global:
                            param.grad.add_(c_global[name].to(self.device) - self.c_local[name].to(self.device))

                optimizer.step()
                num_steps += 1

                preds = outputs.logits.argmax(dim=-1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)

                pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        # Update control variates
        new_params = get_model_params(self.model)
        c_delta = {}
        for name in self.c_local:
            c_new = self.c_local[name] - c_global[name] + (global_params[name] - new_params[name]) / (num_steps * self.lr)
            c_delta[name] = c_new - self.c_local[name]
            self.c_local[name] = c_new

        stats = {
            'client_id': self.client_id,
            'loss': total_loss / max(total_samples, 1),
            'accuracy': total_correct / max(total_samples, 1),
            'samples': total_samples
        }

        return new_params, c_delta, stats
