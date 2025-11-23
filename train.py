#!/usr/bin/env python3
"""
Federated Learning for Text Classification - Non-IID Study

Compares FedAvg vs FedProx on AG News classification with DistilBERT.

Usage:
    python train.py                           # FedAvg with IID data
    python train.py -a fedprox -p topic_skew  # FedProx with non-IID data
    python train.py -a fedprox --alpha 0.01   # FedProx with extreme non-IID
"""

import argparse
import torch
import numpy as np

from src.models.lora_model import (
    get_model_and_tokenizer, get_model_params,
    set_model_params, average_params, get_device
)
from src.data.datasets import load_ag_news, AGNewsDataset, FederatedDataset, SimpleDataset
from src.data.partitioner import NonIIDPartitioner
from src.federated.client import FederatedClient, FedProxClient


# =============================================================================
# DEFAULT PARAMETERS - Edit these for your experiments
# =============================================================================
DEFAULTS = {
    # Algorithm
    'algorithm': 'fedavg',      # 'fedavg' or 'fedprox'
    'partition': 'iid',         # 'iid' or 'topic_skew'

    # Federated Learning
    'num_clients': 5,           # Number of FL clients
    'num_rounds': 5,            # FL communication rounds
    'local_epochs': 2,          # Local training epochs per round

    # Data
    'num_samples': 2000,        # Training samples from AG News
    'batch_size': 32,           # Batch size per client
    'alpha': 0.1,               # Dirichlet alpha (lower = more non-IID)

    # Optimization
    'lr': 2e-5,                 # Learning rate
    'mu': 0.1,                  # FedProx proximal term (only for fedprox)

    # Reproducibility
    'seed': 42,
}
# =============================================================================


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    total_correct = 0
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            preds = outputs.logits.argmax(dim=-1)

            total_correct += (preds == labels).sum().item()
            total_loss += outputs.loss.item() * len(labels)
            total_samples += len(labels)

    return {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples
    }


def compute_divergence(param_list):
    """Compute parameter divergence across clients."""
    if len(param_list) < 2:
        return 0.0
    flat = [torch.cat([p.flatten() for p in params.values()]) for params in param_list]
    stacked = torch.stack(flat)
    mean = stacked.mean(dim=0)
    return torch.mean(torch.norm(stacked - mean, dim=1)).item()


def run_federated(model, clients, test_loader, device, num_rounds):
    """Run federated training (works for both FedAvg and FedProx)."""
    global_params = get_model_params(model)
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'divergence': []
    }

    for round_num in range(num_rounds):
        client_params = []
        client_weights = []
        round_loss, round_acc = 0, 0

        for cid, client in clients.items():
            params, stats = client.train(global_params)
            client_params.append(params)
            client_weights.append(stats['samples'])
            round_loss += stats['loss'] * stats['samples']
            round_acc += stats['accuracy'] * stats['samples']

        # Aggregate
        total_samples = sum(client_weights)
        global_params = average_params(client_params, client_weights)
        set_model_params(model, global_params)

        # Metrics
        divergence = compute_divergence(client_params)
        test_metrics = evaluate(model, test_loader, device)

        history['train_loss'].append(round_loss / total_samples)
        history['train_acc'].append(round_acc / total_samples)
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['divergence'].append(divergence)

        print(f"Round {round_num+1}/{num_rounds} | "
              f"Train Loss: {history['train_loss'][-1]:.3f} | "
              f"Test Acc: {history['test_acc'][-1]:.3f} | "
              f"Divergence: {divergence:.4f}")

    return history


def main():
    parser = argparse.ArgumentParser(
        description='Federated Learning: FedAvg vs FedProx',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-a', '--algorithm', default=DEFAULTS['algorithm'],
                        choices=['fedavg', 'fedprox'], help='FL algorithm')
    parser.add_argument('-p', '--partition', default=DEFAULTS['partition'],
                        choices=['iid', 'topic_skew'], help='Data partition')
    parser.add_argument('-c', '--clients', type=int, default=DEFAULTS['num_clients'],
                        help='Number of clients')
    parser.add_argument('-r', '--rounds', type=int, default=DEFAULTS['num_rounds'],
                        help='FL rounds')
    parser.add_argument('-e', '--local-epochs', type=int, default=DEFAULTS['local_epochs'],
                        help='Local epochs per round')
    parser.add_argument('-n', '--samples', type=int, default=DEFAULTS['num_samples'],
                        help='Training samples')
    parser.add_argument('-b', '--batch', type=int, default=DEFAULTS['batch_size'],
                        help='Batch size')
    parser.add_argument('--alpha', type=float, default=DEFAULTS['alpha'],
                        help='Dirichlet alpha (lower = more non-IID)')
    parser.add_argument('--mu', type=float, default=DEFAULTS['mu'],
                        help='FedProx mu')
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'],
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=DEFAULTS['seed'],
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Print config
    print("=" * 60)
    print("FEDERATED LEARNING EXPERIMENT")
    print("=" * 60)
    print(f"Algorithm:     {args.algorithm.upper()}")
    print(f"Partition:     {args.partition}")
    print(f"Clients:       {args.clients}")
    print(f"Rounds:        {args.rounds}")
    print(f"Local Epochs:  {args.local_epochs}")
    print(f"Samples:       {args.samples}")
    print(f"Alpha:         {args.alpha}" + (" (IID)" if args.partition == 'iid' else " (non-IID)"))
    if args.algorithm == 'fedprox':
        print(f"Mu:            {args.mu}")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice: {device}")

    # Load model
    print("\nLoading DistilBERT...")
    model, tokenizer = get_model_and_tokenizer(num_labels=4)
    model.to(device)

    # Load data
    print(f"\nLoading AG News ({args.samples} samples)...")
    texts, labels = load_ag_news(args.samples, 'train')
    train_dataset = AGNewsDataset(texts, labels, tokenizer, max_length=64)

    # Partition
    print(f"\nPartitioning: {args.partition}")
    raw_data = SimpleDataset({'text': texts, 'label': labels})

    if args.partition == 'iid':
        partitioner = NonIIDPartitioner.create('iid', args.clients)
    else:
        partitioner = NonIIDPartitioner.create('topic_skew', args.clients, alpha=args.alpha)

    partition = partitioner.partition(raw_data)
    print(f"Samples per client: {[len(v) for v in partition.values()]}")

    # Create datasets
    fed_data = FederatedDataset(train_dataset, partition, args.batch)
    test_texts, test_labels = load_ag_news(500, 'test')
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, max_length=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

    # Create clients
    print("\nCreating clients...")
    clients = {}
    for cid in partition.keys():
        if len(partition[cid]) == 0:
            continue
        loader = fed_data.get_client_loader(cid)
        if args.algorithm == 'fedavg':
            clients[cid] = FederatedClient(cid, model, loader, device, args.lr, args.local_epochs)
        else:
            clients[cid] = FedProxClient(cid, model, loader, device, args.lr, args.local_epochs, mu=args.mu)
    print(f"Active clients: {len(clients)}")

    # Train
    print(f"\nStarting {args.algorithm.upper()} training...")
    print("-" * 60)
    history = run_federated(model, clients, test_loader, device, args.rounds)
    print("-" * 60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.3f}")
    print(f"Final Test Loss: {history['test_loss'][-1]:.3f}")


if __name__ == '__main__':
    main()
