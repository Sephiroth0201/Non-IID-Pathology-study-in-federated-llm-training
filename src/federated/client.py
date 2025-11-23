"""
Federated Learning Client

Handles local training on client data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import copy


class FederatedClient:
    """
    Federated Learning Client for local training.

    Each client performs local training on its partition of data
    and reports updates to the server.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 5e-5,
        local_epochs: int = 1,
    ):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs

        # Training statistics
        self.train_loss_history = []
        self.gradient_norms = []

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters (LoRA weights only)."""
        state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                state_dict[name] = param.data.clone()
        return state_dict

    def set_model_params(self, state_dict: Dict[str, torch.Tensor]):
        """Set model parameters from state dict."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.copy_(state_dict[name])

    def compute_gradient_norm(self) -> float:
        """Compute the norm of gradients."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def train(
        self,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform local training.

        Args:
            global_params: Global model parameters to start from
            verbose: Whether to show progress bar

        Returns:
            Tuple of (updated parameters, training statistics)
        """
        if global_params is not None:
            self.set_model_params(global_params)

        self.model.train()
        self.model.to(self.device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

        total_loss = 0.0
        total_steps = 0
        epoch_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            iterator = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}",
                          disable=not verbose)

            for batch in iterator:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                # Track gradient norm
                grad_norm = self.compute_gradient_norm()
                self.gradient_norms.append(grad_norm)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_losses.append(avg_epoch_loss)
            total_loss += epoch_loss
            total_steps += epoch_steps

        # Compute training statistics
        avg_loss = total_loss / max(total_steps, 1)
        self.train_loss_history.append(avg_loss)

        stats = {
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'epoch_losses': epoch_losses,
            'num_samples': len(self.dataloader.dataset),
            'avg_gradient_norm': sum(self.gradient_norms[-total_steps:]) / max(total_steps, 1),
        }

        return self.get_model_params(), stats


class FedProxClient(FederatedClient):
    """
    FedProx Client with proximal term for handling heterogeneity.

    Adds a proximal term to the local objective:
    h_k(w; w^t) = F_k(w) + (mu/2) * ||w - w^t||^2
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 5e-5,
        local_epochs: int = 1,
        mu: float = 0.01,  # Proximal term coefficient
    ):
        super().__init__(client_id, model, dataloader, device, learning_rate, local_epochs)
        self.mu = mu

    def train(
        self,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        FedProx training with proximal term.
        """
        if global_params is not None:
            self.set_model_params(global_params)

        # Store global parameters for proximal term
        global_params_copy = {k: v.clone() for k, v in global_params.items()} if global_params else {}

        self.model.train()
        self.model.to(self.device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

        total_loss = 0.0
        total_prox_loss = 0.0
        total_steps = 0
        epoch_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_prox = 0.0
            epoch_steps = 0

            iterator = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}",
                          disable=not verbose)

            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Add proximal term: (mu/2) * ||w - w^t||^2
                prox_term = 0.0
                if global_params_copy:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in global_params_copy:
                            prox_term += ((param - global_params_copy[name].to(self.device)) ** 2).sum()
                    prox_term = (self.mu / 2) * prox_term

                total_train_loss = loss + prox_term

                total_train_loss.backward()

                grad_norm = self.compute_gradient_norm()
                self.gradient_norms.append(grad_norm)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_prox += prox_term.item() if isinstance(prox_term, torch.Tensor) else prox_term
                epoch_steps += 1

                if verbose:
                    iterator.set_postfix({'loss': loss.item(), 'prox': prox_term.item() if isinstance(prox_term, torch.Tensor) else 0})

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_losses.append(avg_epoch_loss)
            total_loss += epoch_loss
            total_prox_loss += epoch_prox
            total_steps += epoch_steps

        avg_loss = total_loss / max(total_steps, 1)
        self.train_loss_history.append(avg_loss)

        stats = {
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'avg_prox_term': total_prox_loss / max(total_steps, 1),
            'epoch_losses': epoch_losses,
            'num_samples': len(self.dataloader.dataset),
            'avg_gradient_norm': sum(self.gradient_norms[-total_steps:]) / max(total_steps, 1),
        }

        return self.get_model_params(), stats


class SCAFFOLDClient(FederatedClient):
    """
    SCAFFOLD Client with control variates for variance reduction.

    Updates use: y_i = y_i - η_l(g_i(y_i) - c_i + c)
    where c_i is client control variate, c is global control variate.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        learning_rate: float = 5e-5,
        local_epochs: int = 1,
    ):
        super().__init__(client_id, model, dataloader, device, learning_rate, local_epochs)
        self.control_variate = None  # Client control variate c_i

    def initialize_control_variate(self):
        """Initialize client control variate to zeros."""
        self.control_variate = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                self.control_variate[name] = torch.zeros_like(param.data)

    def train(
        self,
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        global_control: Optional[Dict[str, torch.Tensor]] = None,
        verbose: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        SCAFFOLD training with control variates.

        Returns:
            Tuple of (updated params, control variate delta, statistics)
        """
        if global_params is not None:
            self.set_model_params(global_params)

        if self.control_variate is None:
            self.initialize_control_variate()

        if global_control is None:
            global_control = {k: torch.zeros_like(v) for k, v in self.control_variate.items()}

        # Store initial parameters for control variate update
        initial_params = {k: v.clone() for k, v in self.get_model_params().items()}

        self.model.train()
        self.model.to(self.device)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate
        )

        total_loss = 0.0
        total_steps = 0
        epoch_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            iterator = tqdm(self.dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}",
                          disable=not verbose)

            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                loss.backward()

                # SCAFFOLD gradient correction: g - c_i + c
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None and name in self.control_variate:
                        # Correct gradient with control variates
                        correction = global_control[name].to(self.device) - self.control_variate[name].to(self.device)
                        param.grad.data.add_(correction)

                grad_norm = self.compute_gradient_norm()
                self.gradient_norms.append(grad_norm)

                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            epoch_losses.append(avg_epoch_loss)
            total_loss += epoch_loss
            total_steps += epoch_steps

        # Update client control variate
        # c_i^+ = c_i - c + (1/(K*η)) * (x - y)
        # where K = local_epochs, η = learning_rate, x = initial params, y = final params
        final_params = self.get_model_params()
        control_delta = {}

        for name in self.control_variate:
            param_diff = initial_params[name] - final_params[name]
            scale = 1.0 / (self.local_epochs * self.learning_rate * len(self.dataloader))

            # New control variate
            new_ci = self.control_variate[name] - global_control[name] + scale * param_diff

            # Delta to send to server
            control_delta[name] = new_ci - self.control_variate[name]

            # Update local control variate
            self.control_variate[name] = new_ci

        avg_loss = total_loss / max(total_steps, 1)
        self.train_loss_history.append(avg_loss)

        stats = {
            'client_id': self.client_id,
            'avg_loss': avg_loss,
            'epoch_losses': epoch_losses,
            'num_samples': len(self.dataloader.dataset),
            'avg_gradient_norm': sum(self.gradient_norms[-total_steps:]) / max(total_steps, 1),
        }

        return final_params, control_delta, stats
