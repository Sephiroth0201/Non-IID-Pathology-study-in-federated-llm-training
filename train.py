#!/usr/bin/env python3
"""
Federated Learning for Text Classification - Non-IID Study

Model: DistilBERT
Dataset: AG News (4 classes)
Algorithms: FedAvg, FedProx, SCAFFOLD
"""

import argparse
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.models.lora_model import get_model_and_tokenizer, get_model_params, set_model_params, average_params, get_device
from src.data.datasets import load_ag_news, AGNewsDataset, FederatedDataset, SimpleDataset
from src.data.partitioner import NonIIDPartitioner
from src.federated.client import FederatedClient, FedProxClient, SCAFFOLDClient


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


def run_fedavg(model, clients, test_loader, device, num_rounds, participation=1.0):
    """Run FedAvg algorithm."""
    global_params = get_model_params(model)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'divergence': []}

    for round_num in range(num_rounds):
        # Select clients
        num_selected = max(1, int(len(clients) * participation))
        selected = np.random.choice(list(clients.keys()), num_selected, replace=False)

        # Train clients
        client_params = []
        client_weights = []
        round_loss, round_acc = 0, 0

        for cid in selected:
            params, stats = clients[cid].train(global_params)
            client_params.append(params)
            client_weights.append(stats['samples'])
            round_loss += stats['loss'] * stats['samples']
            round_acc += stats['accuracy'] * stats['samples']

        # Aggregate
        total_samples = sum(client_weights)
        global_params = average_params(client_params, client_weights)
        set_model_params(model, global_params)

        # Compute divergence
        divergence = compute_divergence(client_params)

        # Evaluate
        test_metrics = evaluate(model, test_loader, device)

        # Record
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


def run_fedprox(model, clients, test_loader, device, num_rounds, participation=1.0):
    """Run FedProx algorithm."""
    global_params = get_model_params(model)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'divergence': []}

    for round_num in range(num_rounds):
        num_selected = max(1, int(len(clients) * participation))
        selected = np.random.choice(list(clients.keys()), num_selected, replace=False)

        client_params = []
        client_weights = []
        round_loss, round_acc = 0, 0

        for cid in selected:
            params, stats = clients[cid].train(global_params)
            client_params.append(params)
            client_weights.append(stats['samples'])
            round_loss += stats['loss'] * stats['samples']
            round_acc += stats['accuracy'] * stats['samples']

        total_samples = sum(client_weights)
        global_params = average_params(client_params, client_weights)
        set_model_params(model, global_params)

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


def run_scaffold(model, clients, test_loader, device, num_rounds, participation=1.0):
    """Run SCAFFOLD algorithm."""
    global_params = get_model_params(model)
    c_global = {k: torch.zeros_like(v) for k, v in global_params.items()}
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'divergence': []}

    for round_num in range(num_rounds):
        num_selected = max(1, int(len(clients) * participation))
        selected = np.random.choice(list(clients.keys()), num_selected, replace=False)

        client_params = []
        client_weights = []
        c_deltas = []
        round_loss, round_acc = 0, 0

        for cid in selected:
            params, c_delta, stats = clients[cid].train(global_params, c_global)
            client_params.append(params)
            c_deltas.append(c_delta)
            client_weights.append(stats['samples'])
            round_loss += stats['loss'] * stats['samples']
            round_acc += stats['accuracy'] * stats['samples']

        total_samples = sum(client_weights)
        global_params = average_params(client_params, client_weights)
        set_model_params(model, global_params)

        # Update global control
        for key in c_global:
            c_global[key] = c_global[key] + sum(d[key] for d in c_deltas) / len(clients)

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


def compute_divergence(param_list):
    """Compute parameter divergence across clients."""
    if len(param_list) < 2:
        return 0.0
    flat = [torch.cat([p.flatten() for p in params.values()]) for params in param_list]
    stacked = torch.stack(flat)
    mean = stacked.mean(dim=0)
    return torch.mean(torch.norm(stacked - mean, dim=1)).item()


def main():
    parser = argparse.ArgumentParser(description='Federated Learning - Non-IID Study')
    parser.add_argument('-a', '--algorithm', default='fedavg', choices=['fedavg', 'fedprox', 'scaffold'])
    parser.add_argument('-p', '--partition', default='iid', choices=['iid', 'topic_skew'])
    parser.add_argument('-c', '--clients', type=int, default=5)
    parser.add_argument('-r', '--rounds', type=int, default=5)
    parser.add_argument('-n', '--samples', type=int, default=2000)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.1, help='Dirichlet alpha for non-IID')
    parser.add_argument('--mu', type=float, default=0.1, help='FedProx mu')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--local-epochs', type=int, default=2, help='Local epochs per round')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"Federated {args.algorithm.upper()} | {args.partition} | {args.clients} clients | {args.rounds} rounds")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    print("\nLoading DistilBERT...")
    model, tokenizer = get_model_and_tokenizer(num_labels=4)
    model.to(device)

    # Load data
    print(f"\nLoading AG News ({args.samples} samples)...")
    texts, labels = load_ag_news(args.samples, 'train')
    train_dataset = AGNewsDataset(texts, labels, tokenizer, max_length=64)

    # Partition data
    print(f"Partitioning: {args.partition}")
    raw_data = SimpleDataset({'text': texts, 'label': labels})

    if args.partition == 'iid':
        partitioner = NonIIDPartitioner.create('iid', args.clients)
    else:
        partitioner = NonIIDPartitioner.create('topic_skew', args.clients, alpha=args.alpha)

    partition = partitioner.partition(raw_data)
    print(f"Samples per client: {[len(v) for v in partition.values()]}")

    # Create federated dataset
    fed_data = FederatedDataset(train_dataset, partition, args.batch)

    # Create test loader
    test_texts, test_labels = load_ag_news(500, 'test')
    test_dataset = AGNewsDataset(test_texts, test_labels, tokenizer, max_length=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch)

    # Create clients (skip empty ones)
    print("\nCreating clients...")
    clients = {}
    for cid in partition.keys():
        if len(partition[cid]) == 0:
            print(f"  Skipping client {cid} (no samples)")
            continue
        loader = fed_data.get_client_loader(cid)
        if args.algorithm == 'fedavg':
            clients[cid] = FederatedClient(cid, model, loader, device, args.lr, args.local_epochs)
        elif args.algorithm == 'fedprox':
            clients[cid] = FedProxClient(cid, model, loader, device, args.lr, args.local_epochs, mu=args.mu)
        else:
            clients[cid] = SCAFFOLDClient(cid, model, loader, device, args.lr, args.local_epochs)
    print(f"  Active clients: {len(clients)}")

    # Run training
    print(f"\nStarting {args.algorithm.upper()} training...")
    print("-" * 60)

    if args.algorithm == 'fedavg':
        history = run_fedavg(model, clients, test_loader, device, args.rounds)
    elif args.algorithm == 'fedprox':
        history = run_fedprox(model, clients, test_loader, device, args.rounds)
    else:
        history = run_scaffold(model, clients, test_loader, device, args.rounds)

    print("-" * 60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.3f}")
    print(f"Final Test Loss: {history['test_loss'][-1]:.3f}")


if __name__ == '__main__':
    main()
