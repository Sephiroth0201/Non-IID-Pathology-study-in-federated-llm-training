#!/usr/bin/env python3
"""
Run experiments for Non-IID Pathology Study in Federated Learning.

Fast experiment runner for DistilBERT classification on AG News.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_single_experiment(algorithm, partition, num_clients, num_rounds, num_samples,
                          batch_size=32, alpha=0.1, mu=0.01, lr=2e-5, seed=42):
    """Run a single experiment and return results."""
    cmd = [
        sys.executable, 'train.py',
        '-a', algorithm,
        '-p', partition,
        '-c', str(num_clients),
        '-r', str(num_rounds),
        '-n', str(num_samples),
        '-b', str(batch_size),
        '--alpha', str(alpha),
        '--mu', str(mu),
        '--lr', str(lr),
        '--seed', str(seed)
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output
    output = result.stdout + result.stderr
    print(output)

    # Extract final metrics from output
    metrics = parse_output(output)
    return metrics


def parse_output(output):
    """Parse training output to extract metrics."""
    lines = output.split('\n')
    metrics = {
        'rounds': [],
        'train_loss': [],
        'test_acc': [],
        'divergence': [],
        'final_acc': None,
        'final_loss': None
    }

    for line in lines:
        if 'Round' in line and 'Train Loss' in line:
            try:
                # Parse: "Round 1/5 | Train Loss: 1.246 | Test Acc: 0.254 | Divergence: 0.3142"
                parts = line.split('|')
                round_part = parts[0].strip()
                round_num = int(round_part.split()[1].split('/')[0])

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
            except (IndexError, ValueError):
                pass

        if 'Final Test Loss' in line:
            try:
                metrics['final_loss'] = float(line.split(':')[1].strip())
            except (IndexError, ValueError):
                pass

    return metrics


def run_quick_experiments(results_dir='results'):
    """Run quick experiments for testing."""
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        ('fedavg', 'iid'),
        ('fedavg', 'topic_skew'),
        ('fedprox', 'topic_skew'),
        ('scaffold', 'topic_skew'),
    ]

    results = {}
    for algo, partition in experiments:
        name = f"{algo}_{partition}"
        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        metrics = run_single_experiment(
            algorithm=algo,
            partition=partition,
            num_clients=5,
            num_rounds=5,
            num_samples=2000,
            batch_size=32,
            mu=0.1  # Higher mu for FedProx to show effect
        )
        results[name] = metrics

    # Save results
    save_results(results, results_dir, 'quick')
    return results


def run_main_experiments(results_dir='results'):
    """Run full experiment grid."""
    os.makedirs(results_dir, exist_ok=True)

    algorithms = ['fedavg', 'fedprox', 'scaffold']
    partitions = ['iid', 'topic_skew']

    results = {}
    total = len(algorithms) * len(partitions)
    current = 0

    for algo, partition in product(algorithms, partitions):
        current += 1
        name = f"{algo}_{partition}"
        print(f"\n{'='*60}")
        print(f"Experiment {current}/{total}: {name}")
        print(f"{'='*60}")

        metrics = run_single_experiment(
            algorithm=algo,
            partition=partition,
            num_clients=5,
            num_rounds=10,
            num_samples=4000,
            batch_size=32
        )
        results[name] = metrics

    save_results(results, results_dir, 'main')
    return results


def run_failure_experiments(results_dir='results'):
    """Run failure mode experiments."""
    os.makedirs(results_dir, exist_ok=True)

    experiments = [
        ('extreme_skew', {'partition': 'topic_skew', 'alpha': 0.01}),
        ('high_lr', {'partition': 'topic_skew', 'lr': 1e-3}),
        ('many_clients', {'partition': 'topic_skew', 'num_clients': 10}),
    ]

    results = {}
    for name, params in experiments:
        print(f"\n{'='*60}")
        print(f"Failure Mode: {name}")
        print(f"{'='*60}")

        metrics = run_single_experiment(
            algorithm='fedavg',
            partition=params.get('partition', 'topic_skew'),
            num_clients=params.get('num_clients', 5),
            num_rounds=10,
            num_samples=4000,
            alpha=params.get('alpha', 0.1),
            lr=params.get('lr', 2e-5)
        )
        results[name] = metrics

    save_results(results, results_dir, 'failure')
    return results


def save_results(results, results_dir, prefix):
    """Save results to JSON and generate plots."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save JSON
    json_path = os.path.join(results_dir, f'{prefix}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate plots
    generate_plots(results, results_dir, prefix)


def generate_plots(results, results_dir, prefix):
    """Generate analysis plots."""
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Test Accuracy Over Rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in results.items():
        if metrics['test_acc']:
            ax.plot(metrics['rounds'], metrics['test_acc'], marker='o', label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Over Training Rounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_accuracy.png'), dpi=150)
    plt.close()

    # 2. Training Loss Over Rounds
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in results.items():
        if metrics['train_loss']:
            ax.plot(metrics['rounds'], metrics['train_loss'], marker='o', label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Over Rounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_loss.png'), dpi=150)
    plt.close()

    # 3. Client Divergence
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in results.items():
        if metrics['divergence']:
            ax.plot(metrics['rounds'], metrics['divergence'], marker='o', label=name)
    ax.set_xlabel('Round')
    ax.set_ylabel('Client Divergence')
    ax.set_title('Client Parameter Divergence Over Rounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_divergence.png'), dpi=150)
    plt.close()

    # 4. Final Accuracy Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [n for n in results.keys() if results[n]['final_acc'] is not None]
    accs = [results[n]['final_acc'] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylabel('Final Test Accuracy')
    ax.set_title('Final Accuracy Comparison')
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width()/2, acc),
                   xytext=(0, 3), textcoords='offset points', ha='center')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{prefix}_final_accuracy.png'), dpi=150)
    plt.close()

    # 5. Algorithm vs Partition Heatmap
    if len(results) >= 4:
        try:
            algos = sorted(set(n.split('_')[0] for n in results.keys()))
            parts = sorted(set('_'.join(n.split('_')[1:]) for n in results.keys()))

            if len(algos) > 1 and len(parts) > 1:
                data = np.zeros((len(algos), len(parts)))
                for i, algo in enumerate(algos):
                    for j, part in enumerate(parts):
                        key = f"{algo}_{part}"
                        if key in results and results[key]['final_acc'] is not None:
                            data[i, j] = results[key]['final_acc']

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
                ax.set_xticks(range(len(parts)))
                ax.set_yticks(range(len(algos)))
                ax.set_xticklabels(parts)
                ax.set_yticklabels([a.upper() for a in algos])
                ax.set_xlabel('Partition Type')
                ax.set_ylabel('Algorithm')
                ax.set_title('Final Accuracy: Algorithm vs Partition')
                plt.colorbar(im, label='Accuracy')

                for i in range(len(algos)):
                    for j in range(len(parts)):
                        ax.annotate(f'{data[i,j]:.3f}', xy=(j, i),
                                   ha='center', va='center', color='black')

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'{prefix}_heatmap.png'), dpi=150)
                plt.close()
        except Exception as e:
            print(f"Could not generate heatmap: {e}")

    print(f"Plots saved to: {plots_dir}")


def analyze_results(results_dir='results'):
    """Load and analyze existing results."""
    import glob

    json_files = glob.glob(os.path.join(results_dir, '*.json'))
    if not json_files:
        print("No results found to analyze")
        return

    # Load most recent results
    latest = max(json_files, key=os.path.getmtime)
    print(f"Analyzing: {latest}")

    with open(latest, 'r') as f:
        results = json.load(f)

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Experiment':<30} {'Final Acc':<12} {'Final Loss':<12} {'Final Div':<12}")
    print("-"*70)

    for name, metrics in sorted(results.items()):
        acc = metrics.get('final_acc', 'N/A')
        loss = metrics.get('final_loss', 'N/A')
        div = metrics['divergence'][-1] if metrics.get('divergence') else 'N/A'

        acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else acc
        loss_str = f"{loss:.3f}" if isinstance(loss, (int, float)) else loss
        div_str = f"{div:.4f}" if isinstance(div, (int, float)) else div

        print(f"{name:<30} {acc_str:<12} {loss_str:<12} {div_str:<12}")

    print("="*70)

    # Generate plots for existing results
    prefix = os.path.basename(latest).replace('.json', '')
    generate_plots(results, results_dir, prefix)


def main():
    parser = argparse.ArgumentParser(description='Run FL Experiments')
    parser.add_argument('--mode', default='quick',
                       choices=['quick', 'main', 'failure', 'analyze'],
                       help='Experiment mode')
    parser.add_argument('--results-dir', default='results',
                       help='Results directory')

    args = parser.parse_args()

    if args.mode == 'quick':
        print("Running quick experiments (4 experiments, ~5 min)...")
        run_quick_experiments(args.results_dir)

    elif args.mode == 'main':
        print("Running main experiments (6 experiments, ~15 min)...")
        run_main_experiments(args.results_dir)

    elif args.mode == 'failure':
        print("Running failure mode experiments (~10 min)...")
        run_failure_experiments(args.results_dir)

    elif args.mode == 'analyze':
        analyze_results(args.results_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
