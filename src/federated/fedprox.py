"""
FedProx - Federated Proximal Algorithm

Implementation based on:
Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020

Key insight: Add proximal term to handle systems and statistical heterogeneity
h_k(w; w^t) = F_k(w) + (mu/2) * ||w - w^t||^2

The proximal term limits how far client models can drift from global model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from tqdm import tqdm

from .client import FedProxClient
from .server import FedProxServer


class FedProx:
    """
    FedProx Algorithm Implementation.

    Adds proximal regularization to handle heterogeneity.
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        client_dataloaders: Dict[int, DataLoader],
        test_dataloader: DataLoader,
        device: torch.device,
        participation_rate: float = 1.0,
        local_epochs: int = 1,
        learning_rate: float = 5e-5,
        mu: float = 0.01,  # Proximal term coefficient
        num_rounds: int = 50,
        seed: int = 42,
    ):
        """
        Args:
            model: Global model (with LoRA)
            num_clients: Total number of clients
            client_dataloaders: Dict mapping client_id -> DataLoader
            test_dataloader: DataLoader for evaluation
            device: Torch device
            participation_rate: Fraction of clients per round
            local_epochs: Local training epochs per round
            learning_rate: Client learning rate
            mu: Proximal term coefficient (higher = more regularization)
            num_rounds: Total number of FL rounds
            seed: Random seed
        """
        self.model = model
        self.num_clients = num_clients
        self.client_dataloaders = client_dataloaders
        self.test_dataloader = test_dataloader
        self.device = device
        self.participation_rate = participation_rate
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.mu = mu
        self.num_rounds = num_rounds
        self.seed = seed

        # Initialize server
        self.server = FedProxServer(
            global_model=model,
            num_clients=num_clients,
            participation_rate=participation_rate,
            mu=mu,
            seed=seed
        )

        # Create FedProx clients (with proximal term)
        self.clients = {}
        for client_id, dataloader in client_dataloaders.items():
            self.clients[client_id] = FedProxClient(
                client_id=client_id,
                model=model,
                dataloader=dataloader,
                device=device,
                learning_rate=learning_rate,
                local_epochs=local_epochs,
                mu=mu  # Proximal coefficient
            )

        # Training history
        self.history = {
            'round_loss': [],
            'test_loss': [],
            'test_perplexity': [],
            'client_divergence': [],
            'gradient_variance': [],
            'prox_terms': [],  # Track proximal term values
        }

    def _client_train(
        self,
        client_id: int,
        global_params: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Train a single client and return updates."""
        return self.clients[client_id].train(global_params, verbose=False)

    def evaluate(self) -> Tuple[float, float]:
        """Evaluate global model on test data."""
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                num_tokens = attention_mask.sum().item()
                total_loss += outputs.loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = np.exp(min(avg_loss, 100))

        return avg_loss, perplexity

    def compute_client_divergence(
        self,
        client_params: List[Dict[str, torch.Tensor]]
    ) -> float:
        """Compute average divergence between client updates."""
        if len(client_params) < 2:
            return 0.0

        flattened = []
        for params in client_params:
            flat = torch.cat([p.flatten().float() for p in params.values()])
            flattened.append(flat)

        mean = torch.stack(flattened).mean(dim=0)
        divergences = [((f - mean) ** 2).sum().sqrt().item() for f in flattened]
        return np.mean(divergences)

    def compute_gradient_variance(
        self,
        client_stats: List[Dict[str, Any]]
    ) -> float:
        """Compute variance of gradient norms across clients."""
        grad_norms = [s.get('avg_gradient_norm', 0) for s in client_stats]
        return np.var(grad_norms) if grad_norms else 0.0

    def train(
        self,
        eval_every: int = 5,
        verbose: bool = True,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None
    ) -> Dict[str, List[float]]:
        """
        Run federated training with proximal regularization.

        Args:
            eval_every: Evaluate every N rounds
            verbose: Show progress
            callback: Optional callback function(round_num, stats)

        Returns:
            Training history
        """
        iterator = tqdm(range(self.num_rounds), desc="FedProx") if verbose else range(self.num_rounds)

        for round_num in iterator:
            selected_clients = self.server.select_clients()

            client_updates = []
            client_stats = []
            client_weights = []

            for client_id in selected_clients:
                update, stats = self._client_train(client_id, self.server.get_global_params())
                client_updates.append(update)
                client_stats.append(stats)
                client_weights.append(stats['num_samples'])

            # Aggregate (same as FedAvg)
            new_params = self.server.aggregate(client_updates, client_weights)
            self.server.set_global_params(new_params)

            # Record metrics
            avg_loss = np.mean([s['avg_loss'] for s in client_stats])
            self.history['round_loss'].append(avg_loss)

            # Track proximal terms
            avg_prox = np.mean([s.get('avg_prox_term', 0) for s in client_stats])
            self.history['prox_terms'].append(avg_prox)

            divergence = self.compute_client_divergence(client_updates)
            self.history['client_divergence'].append(divergence)

            grad_var = self.compute_gradient_variance(client_stats)
            self.history['gradient_variance'].append(grad_var)

            # Evaluate
            if (round_num + 1) % eval_every == 0 or round_num == self.num_rounds - 1:
                test_loss, perplexity = self.evaluate()
                self.history['test_loss'].append(test_loss)
                self.history['test_perplexity'].append(perplexity)

                if verbose:
                    tqdm.write(f"Round {round_num + 1}: Train Loss={avg_loss:.4f}, "
                              f"Test Loss={test_loss:.4f}, PPL={perplexity:.2f}, "
                              f"Prox={avg_prox:.6f}, Divergence={divergence:.4f}")

            if callback:
                callback(round_num, {
                    'train_loss': avg_loss,
                    'prox_term': avg_prox,
                    'divergence': divergence,
                    'gradient_variance': grad_var,
                    'client_stats': client_stats,
                })

        return self.history

    def get_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.model

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'model_state_dict': self.server.get_global_params(),
            'history': self.history,
            'config': {
                'num_clients': self.num_clients,
                'participation_rate': self.participation_rate,
                'local_epochs': self.local_epochs,
                'learning_rate': self.learning_rate,
                'mu': self.mu,
                'num_rounds': self.num_rounds,
            }
        }, path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.server.set_global_params(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
