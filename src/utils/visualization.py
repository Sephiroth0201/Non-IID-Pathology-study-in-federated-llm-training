"""
Visualization Utilities for Federated Learning Experiments

Generates plots for:
- Training curves (loss, perplexity)
- Client divergence analysis
- Algorithm comparison
- Non-IID impact analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Any


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = 'Training Progress',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot training loss and perplexity curves.

    Args:
        history: Dict with 'round_loss', 'test_loss', 'test_perplexity'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Training loss
    if 'round_loss' in history:
        axes[0].plot(history['round_loss'], label='Train Loss', color='blue')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()

    # Test loss
    if 'test_loss' in history:
        x_vals = np.linspace(0, len(history.get('round_loss', [1])) - 1,
                            len(history['test_loss']))
        axes[1].plot(x_vals, history['test_loss'], label='Test Loss',
                    color='orange', marker='o')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Test Loss')
        axes[1].legend()

    # Perplexity
    if 'test_perplexity' in history:
        x_vals = np.linspace(0, len(history.get('round_loss', [1])) - 1,
                            len(history['test_perplexity']))
        axes[2].plot(x_vals, history['test_perplexity'], label='Perplexity',
                    color='green', marker='o')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Perplexity')
        axes[2].set_title('Test Perplexity')
        axes[2].legend()

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_client_divergence(
    history: Dict[str, List[float]],
    title: str = 'Client Divergence Analysis',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4)
):
    """
    Plot client divergence and gradient variance over rounds.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if 'client_divergence' in history:
        axes[0].plot(history['client_divergence'], color='red')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Divergence')
        axes[0].set_title('Client Update Divergence')
        axes[0].fill_between(range(len(history['client_divergence'])),
                            history['client_divergence'], alpha=0.3, color='red')

    if 'gradient_variance' in history:
        axes[1].plot(history['gradient_variance'], color='purple')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Variance')
        axes[1].set_title('Gradient Norm Variance')
        axes[1].fill_between(range(len(history['gradient_variance'])),
                            history['gradient_variance'], alpha=0.3, color='purple')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_algorithm_comparison(
    results: Dict[str, Dict[str, List[float]]],
    metric: str = 'test_perplexity',
    title: str = 'Algorithm Comparison',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Compare multiple algorithms on a given metric.

    Args:
        results: Dict mapping algorithm_name -> history dict
        metric: Metric to compare
        title: Plot title
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {'fedavg': 'blue', 'fedprox': 'orange', 'scaffold': 'green'}

    for algo_name, history in results.items():
        if metric in history:
            data = history[metric]
            label = algo_name.upper()
            color = colors.get(algo_name.lower(), None)
            ax.plot(data, label=label, color=color, marker='o', markersize=4)

    ax.set_xlabel('Evaluation Point')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_noniid_impact(
    results: Dict[str, Dict[str, float]],
    metric: str = 'final_perplexity',
    title: str = 'Impact of Non-IID Data Distribution',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Bar chart comparing final metrics across non-IID types.

    Args:
        results: Dict mapping partition_type -> summary dict
        metric: Metric to compare
    """
    partitions = list(results.keys())
    values = [results[p].get(metric, 0) for p in partitions]

    fig, ax = plt.subplots(figsize=figsize)

    colors = sns.color_palette('husl', len(partitions))
    bars = ax.bar(partitions, values, color=colors)

    ax.set_xlabel('Partition Type')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords='offset points',
                   ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_participation_rate_impact(
    results: Dict[float, Dict[str, float]],
    metric: str = 'final_perplexity',
    title: str = 'Impact of Client Participation Rate',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
):
    """
    Line plot showing impact of participation rate.
    """
    rates = sorted(results.keys())
    values = [results[r].get(metric, 0) for r in rates]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rates, values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Participation Rate')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Set x-axis as percentages
    ax.set_xticks(rates)
    ax.set_xticklabels([f'{int(r*100)}%' for r in rates])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_heatmap_comparison(
    results: pd.DataFrame,
    title: str = 'Algorithm vs Partition Performance',
    metric: str = 'perplexity',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Heatmap comparing algorithms across partition types.

    Args:
        results: DataFrame with algorithms as rows, partitions as columns
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(results, annot=True, fmt='.2f', cmap='RdYlGn_r',
               ax=ax, cbar_kws={'label': metric.title()})

    ax.set_title(title)
    ax.set_xlabel('Partition Type')
    ax.set_ylabel('Algorithm')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_convergence_analysis(
    histories: Dict[str, Dict[str, List[float]]],
    title: str = 'Convergence Analysis',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Analyze convergence behavior of different experiments.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Training loss curves
    for name, history in histories.items():
        if 'round_loss' in history:
            axes[0].plot(history['round_loss'], label=name, alpha=0.8)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Convergence')
    axes[0].legend(fontsize=8)

    # Divergence over time
    for name, history in histories.items():
        if 'client_divergence' in history:
            axes[1].plot(history['client_divergence'], label=name, alpha=0.8)

    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Client Divergence')
    axes[1].set_title('Client Divergence Over Time')
    axes[1].legend(fontsize=8)

    # Final metrics comparison
    names = list(histories.keys())
    final_losses = []
    for name in names:
        if 'round_loss' in histories[name]:
            final_losses.append(histories[name]['round_loss'][-1])
        else:
            final_losses.append(0)

    axes[2].bar(range(len(names)), final_losses)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Final Training Loss')
    axes[2].set_title('Final Loss Comparison')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_experiment_report(
    results_dir: str,
    output_path: str = 'experiment_report.html'
):
    """
    Generate an HTML report from experiment results.

    Args:
        results_dir: Directory containing experiment results
        output_path: Output HTML file path
    """
    import glob

    # Find all metrics files
    metrics_files = glob.glob(os.path.join(results_dir, '*_metrics.json'))

    all_results = {}
    for filepath in metrics_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            name = os.path.basename(filepath).replace('_metrics.json', '')
            all_results[name] = data

    # Generate HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated LLM Experiment Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            h1 { color: #333; }
            h2 { color: #666; }
            .summary { background-color: #e7f3fe; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Federated LLM Training - Experiment Report</h1>
    """

    # Summary table
    html += "<h2>Experiment Summary</h2>"
    html += "<table><tr><th>Experiment</th><th>Final Loss</th><th>Perplexity</th><th>Rounds</th></tr>"

    for name, data in sorted(all_results.items()):
        summary = data.get('summary', {})
        html += f"""
        <tr>
            <td>{name}</td>
            <td>{summary.get('final_test_loss', 'N/A'):.4f if isinstance(summary.get('final_test_loss'), float) else 'N/A'}</td>
            <td>{summary.get('final_perplexity', 'N/A'):.2f if isinstance(summary.get('final_perplexity'), float) else 'N/A'}</td>
            <td>{summary.get('total_rounds', 'N/A')}</td>
        </tr>
        """

    html += "</table>"

    # Individual experiment details
    for name, data in sorted(all_results.items()):
        html += f"<h2>{name}</h2>"
        html += "<div class='summary'>"

        summary = data.get('summary', {})
        for key, value in summary.items():
            if isinstance(value, float):
                html += f"<p><strong>{key}:</strong> {value:.4f}</p>"
            elif isinstance(value, dict):
                html += f"<p><strong>{key}:</strong> {json.dumps(value)}</p>"
            else:
                html += f"<p><strong>{key}:</strong> {value}</p>"

        html += "</div>"

    html += "</body></html>"

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Report saved to: {output_path}")
    return output_path


def load_experiment_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all experiment results from a directory.
    """
    import glob

    results = {}
    for filepath in glob.glob(os.path.join(results_dir, '*_metrics.json')):
        with open(filepath, 'r') as f:
            data = json.load(f)
            name = os.path.basename(filepath).replace('_metrics.json', '')
            results[name] = data

    return results
