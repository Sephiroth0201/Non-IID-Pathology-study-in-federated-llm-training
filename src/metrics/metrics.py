"""
Metrics Module for Federated LLM Training

Implements various metrics for evaluating:
1. Language quality (perplexity, validation loss)
2. Training stability (gradient norm variance, client divergence)
3. Representation quality (cosine similarity, attention entropy)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json
import os


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    PPL = exp(loss), clipped to avoid overflow.
    """
    return np.exp(min(loss, 100))


def compute_gradient_divergence(
    gradients: List[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    Compute gradient divergence metrics across clients.

    Returns:
        Dict with 'mean_norm', 'variance', 'max_divergence'
    """
    if not gradients:
        return {'mean_norm': 0.0, 'variance': 0.0, 'max_divergence': 0.0}

    # Flatten gradients
    flattened = []
    for grad_dict in gradients:
        flat = torch.cat([g.flatten().float() for g in grad_dict.values()])
        flattened.append(flat)

    # Stack for vectorized operations
    stacked = torch.stack(flattened)

    # Mean gradient
    mean_grad = stacked.mean(dim=0)

    # Norms
    norms = [f.norm().item() for f in flattened]
    mean_norm = np.mean(norms)

    # Variance of norms
    variance = np.var(norms)

    # Max divergence from mean
    divergences = [(f - mean_grad).norm().item() for f in flattened]
    max_divergence = max(divergences)

    return {
        'mean_norm': mean_norm,
        'variance': variance,
        'max_divergence': max_divergence
    }


def compute_weight_drift(
    current_weights: Dict[str, torch.Tensor],
    reference_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute weight drift metrics between two sets of weights.

    Returns:
        Dict with 'l2_drift', 'cosine_similarity', 'relative_drift'
    """
    # L2 drift
    l2_drift = 0.0
    total_norm = 0.0

    for key in current_weights:
        if key in reference_weights:
            diff = current_weights[key].float() - reference_weights[key].float()
            l2_drift += (diff ** 2).sum().item()
            total_norm += (reference_weights[key].float() ** 2).sum().item()

    l2_drift = l2_drift ** 0.5
    relative_drift = l2_drift / max(total_norm ** 0.5, 1e-8)

    # Cosine similarity
    vec1 = torch.cat([current_weights[k].flatten().float() for k in sorted(current_weights.keys())])
    vec2 = torch.cat([reference_weights[k].flatten().float() for k in sorted(reference_weights.keys())])

    cosine_sim = torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()

    return {
        'l2_drift': l2_drift,
        'cosine_similarity': cosine_sim,
        'relative_drift': relative_drift
    }


def compute_client_update_statistics(
    client_updates: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int]
) -> Dict[str, float]:
    """
    Compute statistics about client updates.

    Returns:
        Dict with various update statistics
    """
    if not client_updates:
        return {}

    # Flatten updates
    flattened = []
    for update in client_updates:
        flat = torch.cat([u.flatten().float() for u in update.values()])
        flattened.append(flat)

    stacked = torch.stack(flattened)

    # Weighted mean
    weights = np.array(client_data_sizes) / sum(client_data_sizes)
    weighted_mean = sum(w * f for w, f in zip(weights, flattened))

    # Statistics
    update_norms = [f.norm().item() for f in flattened]
    divergences = [(f - weighted_mean).norm().item() for f in flattened]

    return {
        'mean_update_norm': np.mean(update_norms),
        'std_update_norm': np.std(update_norms),
        'mean_divergence': np.mean(divergences),
        'max_divergence': max(divergences),
        'min_divergence': min(divergences),
    }


class MetricsCollector:
    """
    Centralized metrics collection and logging for federated training.
    """

    def __init__(self, log_dir: str = 'results'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.metrics = defaultdict(list)
        self.round_metrics = []
        self.client_metrics = defaultdict(lambda: defaultdict(list))

    def log_round(
        self,
        round_num: int,
        train_loss: float,
        test_loss: Optional[float] = None,
        test_perplexity: Optional[float] = None,
        client_divergence: Optional[float] = None,
        gradient_variance: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics for a training round."""
        round_data = {
            'round': round_num,
            'train_loss': train_loss,
        }

        if test_loss is not None:
            round_data['test_loss'] = test_loss
        if test_perplexity is not None:
            round_data['test_perplexity'] = test_perplexity
        if client_divergence is not None:
            round_data['client_divergence'] = client_divergence
        if gradient_variance is not None:
            round_data['gradient_variance'] = gradient_variance

        if additional_metrics:
            round_data.update(additional_metrics)

        self.round_metrics.append(round_data)

        # Update time series
        for key, value in round_data.items():
            if key != 'round':
                self.metrics[key].append(value)

    def log_client(
        self,
        round_num: int,
        client_id: int,
        loss: float,
        gradient_norm: float,
        num_samples: int,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics for a specific client."""
        self.client_metrics[client_id]['loss'].append(loss)
        self.client_metrics[client_id]['gradient_norm'].append(gradient_norm)
        self.client_metrics[client_id]['num_samples'].append(num_samples)

        if additional_metrics:
            for key, value in additional_metrics.items():
                self.client_metrics[client_id][key].append(value)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of training."""
        summary = {
            'total_rounds': len(self.round_metrics),
        }

        # Final metrics
        if self.round_metrics:
            final = self.round_metrics[-1]
            summary['final_train_loss'] = final.get('train_loss')
            summary['final_test_loss'] = final.get('test_loss')
            summary['final_perplexity'] = final.get('test_perplexity')

        # Best metrics
        if 'test_loss' in self.metrics and self.metrics['test_loss']:
            summary['best_test_loss'] = min(self.metrics['test_loss'])

        if 'test_perplexity' in self.metrics and self.metrics['test_perplexity']:
            summary['best_perplexity'] = min(self.metrics['test_perplexity'])

        # Convergence metrics
        if 'train_loss' in self.metrics and len(self.metrics['train_loss']) > 10:
            # Average of last 10 rounds
            summary['converged_train_loss'] = np.mean(self.metrics['train_loss'][-10:])
            # Stability (std of last 10 rounds)
            summary['convergence_stability'] = np.std(self.metrics['train_loss'][-10:])

        # Client heterogeneity
        if self.client_metrics:
            client_final_losses = []
            for client_id, metrics in self.client_metrics.items():
                if metrics['loss']:
                    client_final_losses.append(metrics['loss'][-1])

            if client_final_losses:
                summary['client_loss_variance'] = np.var(client_final_losses)
                summary['client_loss_range'] = max(client_final_losses) - min(client_final_losses)

        return summary

    def save(self, filename: str = 'metrics.json'):
        """Save metrics to JSON file."""
        filepath = os.path.join(self.log_dir, filename)

        data = {
            'round_metrics': self.round_metrics,
            'summary': self.get_summary(),
            'client_metrics': {
                str(k): dict(v) for k, v in self.client_metrics.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return filepath

    def load(self, filename: str = 'metrics.json'):
        """Load metrics from JSON file."""
        filepath = os.path.join(self.log_dir, filename)

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.round_metrics = data['round_metrics']
        self.client_metrics = defaultdict(lambda: defaultdict(list))
        for k, v in data.get('client_metrics', {}).items():
            self.client_metrics[int(k)] = defaultdict(list, v)

        # Rebuild metrics time series
        self.metrics = defaultdict(list)
        for round_data in self.round_metrics:
            for key, value in round_data.items():
                if key != 'round':
                    self.metrics[key].append(value)


class FailureModeDetector:
    """
    Detect failure modes in federated training.

    Monitors for:
    - Divergence (loss increasing)
    - Oscillation (loss fluctuating)
    - Client drift (high divergence)
    - Slow convergence
    """

    def __init__(
        self,
        divergence_threshold: float = 0.1,
        oscillation_window: int = 10,
        drift_threshold: float = 0.5,
        convergence_window: int = 20
    ):
        self.divergence_threshold = divergence_threshold
        self.oscillation_window = oscillation_window
        self.drift_threshold = drift_threshold
        self.convergence_window = convergence_window

        self.loss_history = []
        self.divergence_history = []

    def update(self, loss: float, client_divergence: float):
        """Update with new metrics."""
        self.loss_history.append(loss)
        self.divergence_history.append(client_divergence)

    def detect_divergence(self) -> bool:
        """Check if training is diverging."""
        if len(self.loss_history) < 5:
            return False

        recent = self.loss_history[-5:]
        # Check if loss is consistently increasing
        increasing = all(recent[i] > recent[i-1] for i in range(1, len(recent)))
        # Check magnitude of increase
        relative_increase = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-8)

        return increasing and relative_increase > self.divergence_threshold

    def detect_oscillation(self) -> bool:
        """Check if training is oscillating."""
        if len(self.loss_history) < self.oscillation_window:
            return False

        recent = self.loss_history[-self.oscillation_window:]
        # Count direction changes
        changes = sum(
            1 for i in range(1, len(recent) - 1)
            if (recent[i] - recent[i-1]) * (recent[i+1] - recent[i]) < 0
        )

        # High number of direction changes indicates oscillation
        return changes > self.oscillation_window * 0.6

    def detect_client_drift(self) -> bool:
        """Check if clients are drifting apart."""
        if not self.divergence_history:
            return False

        # Check if divergence is above threshold
        return self.divergence_history[-1] > self.drift_threshold

    def detect_slow_convergence(self) -> bool:
        """Check if convergence is too slow."""
        if len(self.loss_history) < self.convergence_window:
            return False

        window = self.loss_history[-self.convergence_window:]
        # Check if loss hasn't decreased significantly
        improvement = (window[0] - window[-1]) / max(abs(window[0]), 1e-8)

        return improvement < 0.01  # Less than 1% improvement

    def get_status(self) -> Dict[str, Any]:
        """Get current failure detection status."""
        return {
            'diverging': self.detect_divergence(),
            'oscillating': self.detect_oscillation(),
            'client_drift': self.detect_client_drift(),
            'slow_convergence': self.detect_slow_convergence(),
            'current_loss': self.loss_history[-1] if self.loss_history else None,
            'current_divergence': self.divergence_history[-1] if self.divergence_history else None,
        }

    def get_recommendations(self) -> List[str]:
        """Get recommendations based on detected issues."""
        recommendations = []

        if self.detect_divergence():
            recommendations.append("Training is diverging. Try reducing learning rate or using FedProx with higher mu.")

        if self.detect_oscillation():
            recommendations.append("Training is oscillating. Consider increasing local epochs or using SCAFFOLD.")

        if self.detect_client_drift():
            recommendations.append("High client drift detected. Consider using FedProx or SCAFFOLD, or increase participation rate.")

        if self.detect_slow_convergence():
            recommendations.append("Slow convergence. Consider increasing learning rate or reducing non-IID severity.")

        if not recommendations:
            recommendations.append("Training appears healthy.")

        return recommendations
