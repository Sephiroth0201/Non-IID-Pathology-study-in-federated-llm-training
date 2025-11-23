#!/usr/bin/env python3
"""
Experiment Runner for Non-IID Federated Learning Study

Runs systematic experiments comparing FedAvg vs FedProx under different
data heterogeneity conditions.

Usage:
    python run_experiments.py --mode quick     # Quick test (~5 min)
    python run_experiments.py --mode main      # Full experiments (~15 min)
    python run_experiments.py --mode failure   # Failure mode tests
    python run_experiments.py --mode analyze   # Analyze existing results
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# EXPERIMENT CONFIGURATIONS - Edit these for your experiments
# =============================================================================

# Quick experiments (for testing)
QUICK_EXPERIMENTS = [
    {'name': 'fedavg_iid', 'algorithm': 'fedavg', 'partition': 'iid'},
    {'name': 'fedavg_noniid', 'algorithm': 'fedavg', 'partition': 'topic_skew'},
    {'name': 'fedprox_noniid', 'algorithm': 'fedprox', 'partition': 'topic_skew'},
]

# Main experiments (full comparison)
MAIN_EXPERIMENTS = [
    {'name': 'fedavg_iid', 'algorithm': 'fedavg', 'partition': 'iid'},
    {'name': 'fedavg_noniid', 'algorithm': 'fedavg', 'partition': 'topic_skew'},
    {'name': 'fedprox_iid', 'algorithm': 'fedprox', 'partition': 'iid'},
    {'name': 'fedprox_noniid', 'algorithm': 'fedprox', 'partition': 'topic_skew'},
]

# Failure mode experiments
FAILURE_EXPERIMENTS = [
    {'name': 'extreme_noniid', 'algorithm': 'fedavg', 'partition': 'topic_skew', 'alpha': 0.01},
    {'name': 'fedprox_extreme', 'algorithm': 'fedprox', 'partition': 'topic_skew', 'alpha': 0.01},
    {'name': 'high_lr', 'algorithm': 'fedavg', 'partition': 'topic_skew', 'lr': 1e-3},
    {'name': 'many_clients', 'algorithm': 'fedavg', 'partition': 'topic_skew', 'clients': 10},
]

# Default parameters for all experiments
DEFAULT_PARAMS = {
    'clients': 5,
    'rounds': 5,
    'samples': 2000,
    'batch': 32,
    'alpha': 0.1,
    'mu': 0.1,
    'lr': 2e-5,
    'local_epochs': 2,
    'seed': 42,
}

# =============================================================================


def run_experiment(config):
    """Run a single experiment and return metrics."""
    params = {**DEFAULT_PARAMS, **config}

    cmd = [
        sys.executable, 'train.py',
        '-a', params['algorithm'],
        '-p', params['partition'],
        '-c', str(params['clients']),
        '-r', str(params['rounds']),
        '-n', str(params['samples']),
        '-b', str(params['batch']),
        '-e', str(params['local_epochs']),
        '--alpha', str(params['alpha']),
        '--mu', str(params['mu']),
        '--lr', str(params['lr']),
        '--seed', str(params['seed']),
    ]

    print(f"\n>>> {params['name']}")
    print(f"    Command: {' '.join(cmd[1:])}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output)

    return parse_output(output)


def parse_output(output):
    """Parse training output to extract metrics."""
    metrics = {
        'rounds': [],
        'train_loss': [],
        'test_acc': [],
        'divergence': [],
        'final_acc': None,
        'final_loss': None
    }

    for line in output.split('\n'):
        if 'Round' in line and 'Train Loss' in line:
            try:
                parts = line.split('|')
                round_num = int(parts[0].strip().split()[1].split('/')[0])
                train_loss = float(parts[1].split(':')[1].strip())
                test_acc = float(parts[2].split(':')[1].strip())
                div = float(parts[3].split(':')[1].strip())

                metrics['rounds'].append(round_num)
                metrics['train_loss'].append(train_loss)
                metrics['test_acc'].append(test_acc)
                metrics['divergence'].append(div)
            except (IndexError, ValueError):
                continue

        if 'Final Test Accuracy' in line:
            try:
                metrics['final_acc'] = float(line.split(':')[1].strip())
            except:
                pass

        if 'Final Test Loss' in line:
            try:
                metrics['final_loss'] = float(line.split(':')[1].strip())
            except:
                pass

    return metrics


def run_experiments(experiments, results_dir, prefix):
    """Run a set of experiments."""
    os.makedirs(results_dir, exist_ok=True)
    results = {}

    print("=" * 60)
    print(f"Running {len(experiments)} experiments")
    print("=" * 60)

    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['name']}")
        results[exp['name']] = run_experiment(exp)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(results_dir, f'{prefix}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Generate plots
    generate_plots(results, results_dir, prefix)

    # Print summary
    print_summary(results)

    return results


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Final Acc':<12} {'Final Loss':<12} {'Divergence':<12}")
    print("-" * 70)

    for name, m in sorted(results.items()):
        acc = f"{m['final_acc']:.3f}" if m['final_acc'] else "N/A"
        loss = f"{m['final_loss']:.3f}" if m['final_loss'] else "N/A"
        div = f"{m['divergence'][-1]:.4f}" if m['divergence'] else "N/A"
        print(f"{name:<25} {acc:<12} {loss:<12} {div:<12}")

    print("=" * 70)


def generate_plots(results, results_dir, prefix):
    """Generate analysis plots."""
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    valid_results = {k: v for k, v in results.items() if v['test_acc']}
    if not valid_results:
        print("No valid results to plot")
        return

    # 1. Test Accuracy Over Rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, m in valid_results.items():
        ax.plot(m['rounds'], m['test_acc'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy Over Training Rounds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_accuracy.png'), dpi=150)
    plt.close()

    # 2. Training Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, m in valid_results.items():
        ax.plot(m['rounds'], m['train_loss'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Over Rounds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_loss.png'), dpi=150)
    plt.close()

    # 3. Client Divergence
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, m in valid_results.items():
        ax.plot(m['rounds'], m['divergence'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Client Divergence', fontsize=12)
    ax.set_title('Client Parameter Divergence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_divergence.png'), dpi=150)
    plt.close()

    # 4. Final Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(valid_results.keys())
    accs = [valid_results[n]['final_acc'] or 0 for n in names]
    colors = ['#2ecc71' if 'fedprox' in n else '#3498db' for n in names]
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel('Final Test Accuracy', fontsize=12)
    ax.set_title('Final Accuracy Comparison', fontsize=14)
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_final_accuracy.png'), dpi=150)
    plt.close()

    print(f"Plots saved: {plots_dir}/")


def analyze_results(results_dir):
    """Analyze existing results."""
    import glob

    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    if not json_files:
        print("No results found")
        return

    latest = json_files[-1]
    print(f"Analyzing: {latest}")

    with open(latest, 'r') as f:
        results = json.load(f)

    print_summary(results)
    prefix = os.path.basename(latest).replace('.json', '')
    generate_plots(results, results_dir, f"analysis_{prefix}")


def main():
    parser = argparse.ArgumentParser(
        description='Run FL experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mode', default='quick',
                       choices=['quick', 'main', 'failure', 'analyze'],
                       help='Experiment mode')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory')

    args = parser.parse_args()

    if args.mode == 'quick':
        print("Running QUICK experiments (~5 min)...")
        run_experiments(QUICK_EXPERIMENTS, args.results_dir, 'quick')

    elif args.mode == 'main':
        print("Running MAIN experiments (~15 min)...")
        run_experiments(MAIN_EXPERIMENTS, args.results_dir, 'main')

    elif args.mode == 'failure':
        print("Running FAILURE MODE experiments...")
        run_experiments(FAILURE_EXPERIMENTS, args.results_dir, 'failure')

    elif args.mode == 'analyze':
        analyze_results(args.results_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
