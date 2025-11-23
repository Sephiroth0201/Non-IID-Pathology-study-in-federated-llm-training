"""
Federated Learning Server

Coordinates federated training by:
1. Selecting participating clients each round
2. Broadcasting global model to clients
3. Aggregating client updates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import random
from abc import ABC, abstractmethod


class FederatedServer(ABC):
    """
    Abstract base class for federated learning servers.
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        participation_rate: float = 1.0,
        seed: int = 42,
    ):
        self.global_model = global_model
        self.num_clients = num_clients
        self.participation_rate = participation_rate
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)

        # Training history
        self.round_history = []
        self.global_params_history = []

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters (LoRA weights)."""
        state_dict = {}
        for name, param in self.global_model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                state_dict[name] = param.data.clone()
        return state_dict

    def set_global_params(self, state_dict: Dict[str, torch.Tensor]):
        """Set global model parameters."""
        model_state = self.global_model.state_dict()
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)

    def select_clients(self) -> List[int]:
        """Select clients to participate in this round."""
        num_selected = max(1, int(self.num_clients * self.participation_rate))
        selected = random.sample(range(self.num_clients), num_selected)
        return selected

    @abstractmethod
    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates into new global parameters.

        Args:
            client_updates: List of parameter updates from clients
            client_weights: Optional weights for weighted averaging

        Returns:
            Aggregated parameters
        """
        pass

    def run_round(
        self,
        client_train_fn: Callable[[int, Dict[str, torch.Tensor]], Tuple[Dict[str, torch.Tensor], Dict[str, Any]]],
        selected_clients: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Execute one round of federated learning.

        Args:
            client_train_fn: Function that trains a client, takes (client_id, global_params)
            selected_clients: Optional list of clients to use (otherwise select randomly)

        Returns:
            Round statistics
        """
        if selected_clients is None:
            selected_clients = self.select_clients()

        global_params = self.get_global_params()

        # Train selected clients
        client_updates = []
        client_stats = []
        client_weights = []

        for client_id in selected_clients:
            update, stats = client_train_fn(client_id, global_params)
            client_updates.append(update)
            client_stats.append(stats)
            client_weights.append(stats.get('num_samples', 1))

        # Aggregate updates
        new_global_params = self.aggregate(client_updates, client_weights)
        self.set_global_params(new_global_params)

        # Compile round statistics
        round_stats = {
            'selected_clients': selected_clients,
            'num_clients': len(selected_clients),
            'avg_client_loss': np.mean([s['avg_loss'] for s in client_stats]),
            'client_stats': client_stats,
        }

        self.round_history.append(round_stats)
        self.global_params_history.append(self.get_global_params())

        return round_stats


class FedAvgServer(FederatedServer):
    """
    FedAvg Server - Simple weighted averaging of client updates.

    w^{t+1} = sum_k (n_k / n) * w_k^{t+1}
    """

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation: weighted average of client parameters."""
        if not client_updates:
            return self.get_global_params()

        if client_weights is None:
            client_weights = [1.0] * len(client_updates)

        # Normalize weights
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Weighted average
        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = sum(
                w * update[key].float() for w, update in zip(normalized_weights, client_updates)
            )

        return aggregated


class FedProxServer(FederatedServer):
    """
    FedProx Server - Same aggregation as FedAvg.

    The proximal term is handled client-side.
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        participation_rate: float = 1.0,
        mu: float = 0.01,
        seed: int = 42,
    ):
        super().__init__(global_model, num_clients, participation_rate, seed)
        self.mu = mu

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """FedProx uses same aggregation as FedAvg."""
        if not client_updates:
            return self.get_global_params()

        if client_weights is None:
            client_weights = [1.0] * len(client_updates)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = sum(
                w * update[key].float() for w, update in zip(normalized_weights, client_updates)
            )

        return aggregated


class SCAFFOLDServer(FederatedServer):
    """
    SCAFFOLD Server with global control variate.

    Maintains global control variate c that helps reduce variance
    in client updates.
    """

    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        participation_rate: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(global_model, num_clients, participation_rate, seed)
        self.global_control = None

    def initialize_control_variate(self):
        """Initialize global control variate to zeros."""
        self.global_control = {}
        for name, param in self.global_model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                self.global_control[name] = torch.zeros_like(param.data)

    def get_global_control(self) -> Dict[str, torch.Tensor]:
        """Get global control variate."""
        if self.global_control is None:
            self.initialize_control_variate()
        return {k: v.clone() for k, v in self.global_control.items()}

    def aggregate(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Standard weighted averaging for parameters."""
        if not client_updates:
            return self.get_global_params()

        if client_weights is None:
            client_weights = [1.0] * len(client_updates)

        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        aggregated = {}
        for key in client_updates[0].keys():
            aggregated[key] = sum(
                w * update[key].float() for w, update in zip(normalized_weights, client_updates)
            )

        return aggregated

    def aggregate_control_variates(
        self,
        control_deltas: List[Dict[str, torch.Tensor]],
        num_total_clients: int
    ):
        """
        Update global control variate with client deltas.

        c = c + (1/N) * sum_i (c_i^+ - c_i)
        """
        if self.global_control is None:
            self.initialize_control_variate()

        num_participating = len(control_deltas)

        for key in self.global_control:
            delta_sum = sum(delta[key] for delta in control_deltas)
            self.global_control[key] = self.global_control[key] + (num_participating / num_total_clients) * delta_sum

    def run_scaffold_round(
        self,
        client_train_fn: Callable,
        selected_clients: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Execute one round of SCAFFOLD training.

        client_train_fn should return (params, control_delta, stats)
        """
        if selected_clients is None:
            selected_clients = self.select_clients()

        global_params = self.get_global_params()
        global_control = self.get_global_control()

        client_updates = []
        control_deltas = []
        client_stats = []
        client_weights = []

        for client_id in selected_clients:
            update, control_delta, stats = client_train_fn(
                client_id, global_params, global_control
            )
            client_updates.append(update)
            control_deltas.append(control_delta)
            client_stats.append(stats)
            client_weights.append(stats.get('num_samples', 1))

        # Aggregate parameters
        new_global_params = self.aggregate(client_updates, client_weights)
        self.set_global_params(new_global_params)

        # Aggregate control variates
        self.aggregate_control_variates(control_deltas, self.num_clients)

        round_stats = {
            'selected_clients': selected_clients,
            'num_clients': len(selected_clients),
            'avg_client_loss': np.mean([s['avg_loss'] for s in client_stats]),
            'client_stats': client_stats,
        }

        self.round_history.append(round_stats)
        self.global_params_history.append(self.get_global_params())

        return round_stats
