#!/usr/bin/env python3
"""
Run systematic experiments for Non-IID Pathology Study.

This script runs a comprehensive set of experiments to analyze
how federated LLM training fails under different non-IID conditions.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train import run_experiment, set_seed
from src.utils.config import (
    ExperimentConfig, ModelConfig, DataConfig, FederatedConfig,
    create_experiment_configs, create_failure_mode_configs
)
from src.utils.visualization import (
    plot_algorithm_comparison,
    plot_noniid_impact,
    plot_convergence_analysis,
    plot_heatmap_comparison,
    create_experiment_report,
    load_experiment_results
)
import pandas as pd


def run_main_experiments(base_config: ExperimentConfig, quick: bool = False):
    """
    Run main experiment grid: Algorithm x Partition x Participation Rate

    Args:
        base_config: Base configuration to modify
        quick: If True, run fewer rounds for testing
    """
    algorithms = ['fedavg', 'fedprox', 'scaffold']
    partitions = ['iid', 'topic_skew', 'style_skew', 'token_skew']
    participation_rates = [1.0] if quick else [0.3, 0.5, 1.0]

    num_rounds = 10 if quick else base_config.federated.num_rounds

    results = {}
    total = len(algorithms) * len(partitions) * len(participation_rates)
    current = 0

    for algo, partition, rate in product(algorithms, partitions, participation_rates):
        current += 1
        exp_name = f"{algo}_{partition}_pr{int(rate*100)}"
        print(f"\n{'='*60}")
        print(f"Experiment {current}/{total}: {exp_name}")
        print(f"{'='*60}")

        config = ExperimentConfig(
            name=exp_name,
            model=ModelConfig(
                name=base_config.model.name,
                lora_r=base_config.model.lora_r,
                lora_alpha=base_config.model.lora_alpha,
                lora_dropout=base_config.model.lora_dropout,
                max_length=base_config.model.max_length
            ),
            data=DataConfig(
                dataset=base_config.data.dataset,
                partition_strategy=partition,
                num_clients=base_config.data.num_clients,
                alpha=base_config.data.alpha,
                batch_size=base_config.data.batch_size,
                seed=base_config.seed
            ),
            federated=FederatedConfig(
                algorithm=algo,
                num_rounds=num_rounds,
                participation_rate=rate,
                local_epochs=base_config.federated.local_epochs,
                learning_rate=base_config.federated.learning_rate,
                mu=base_config.federated.mu
            ),
            eval_every=base_config.eval_every,
            save_checkpoints=base_config.save_checkpoints,
            results_dir=base_config.results_dir,
            seed=base_config.seed,
            device=base_config.device
        )

        try:
            history, summary = run_experiment(config)
            results[exp_name] = {
                'history': history,
                'summary': summary,
                'config': {
                    'algorithm': algo,
                    'partition': partition,
                    'participation_rate': rate
                }
            }
        except Exception as e:
            print(f"Error in {exp_name}: {e}")
            results[exp_name] = {'error': str(e)}

    return results


def run_failure_mode_experiments(base_config: ExperimentConfig):
    """
    Run failure mode experiments:
    - Extreme topic isolation
    - Very low participation
    - High learning rate
    - Many local epochs
    """
    experiments = [
        ('extreme_isolation', {'partition_strategy': 'topic_skew', 'alpha': 0.01}),
        ('low_participation', {'participation_rate': 0.1}),
        ('high_lr', {'learning_rate': 1e-3}),
        ('many_epochs', {'local_epochs': 5}),
    ]

    results = {}

    for exp_name, overrides in experiments:
        print(f"\n{'='*60}")
        print(f"Failure Mode: {exp_name}")
        print(f"{'='*60}")

        # Create config with overrides
        data_config = DataConfig(
            dataset=base_config.data.dataset,
            partition_strategy=overrides.get('partition_strategy', base_config.data.partition_strategy),
            num_clients=base_config.data.num_clients,
            alpha=overrides.get('alpha', base_config.data.alpha),
            batch_size=base_config.data.batch_size,
            seed=base_config.seed
        )

        fed_config = FederatedConfig(
            algorithm='fedavg',  # Test with FedAvg to see failures
            num_rounds=base_config.federated.num_rounds,
            participation_rate=overrides.get('participation_rate', base_config.federated.participation_rate),
            local_epochs=overrides.get('local_epochs', base_config.federated.local_epochs),
            learning_rate=overrides.get('learning_rate', base_config.federated.learning_rate),
            mu=base_config.federated.mu
        )

        config = ExperimentConfig(
            name=f"failure_{exp_name}",
            model=base_config.model,
            data=data_config,
            federated=fed_config,
            eval_every=base_config.eval_every,
            results_dir=base_config.results_dir,
            seed=base_config.seed,
            device=base_config.device
        )

        try:
            history, summary = run_experiment(config)
            results[exp_name] = {
                'history': history,
                'summary': summary
            }
        except Exception as e:
            print(f"Error in {exp_name}: {e}")
            results[exp_name] = {'error': str(e)}

    return results


def run_algorithm_comparison(base_config: ExperimentConfig, partition: str = 'topic_skew'):
    """
    Compare all algorithms on a specific partition type.
    """
    algorithms = ['fedavg', 'fedprox', 'scaffold']
    results = {}

    for algo in algorithms:
        exp_name = f"compare_{algo}_{partition}"
        print(f"\n{'='*60}")
        print(f"Algorithm Comparison: {exp_name}")
        print(f"{'='*60}")

        config = ExperimentConfig(
            name=exp_name,
            model=base_config.model,
            data=DataConfig(
                dataset=base_config.data.dataset,
                partition_strategy=partition,
                num_clients=base_config.data.num_clients,
                alpha=base_config.data.alpha,
                batch_size=base_config.data.batch_size,
                seed=base_config.seed
            ),
            federated=FederatedConfig(
                algorithm=algo,
                num_rounds=base_config.federated.num_rounds,
                participation_rate=base_config.federated.participation_rate,
                local_epochs=base_config.federated.local_epochs,
                learning_rate=base_config.federated.learning_rate,
                mu=base_config.federated.mu
            ),
            eval_every=base_config.eval_every,
            results_dir=base_config.results_dir,
            seed=base_config.seed,
            device=base_config.device
        )

        try:
            history, summary = run_experiment(config)
            results[algo] = history
        except Exception as e:
            print(f"Error in {algo}: {e}")

    return results


def generate_analysis(results_dir: str):
    """
    Generate analysis plots from experiment results.
    """
    print("\nGenerating analysis plots...")

    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    # Load results
    results = load_experiment_results(results_dir)

    if not results:
        print("No results found to analyze")
        return

    # Algorithm comparison plot
    algo_results = {}
    for name, data in results.items():
        for algo in ['fedavg', 'fedprox', 'scaffold']:
            if algo in name.lower():
                if algo not in algo_results:
                    algo_results[algo] = {}
                if 'round_metrics' in data:
                    history = {
                        'test_perplexity': [r.get('test_perplexity') for r in data['round_metrics']
                                           if r.get('test_perplexity') is not None]
                    }
                    algo_results[algo] = history

    if algo_results:
        plot_algorithm_comparison(
            algo_results,
            metric='test_perplexity',
            title='Algorithm Comparison - Perplexity',
            save_path=os.path.join(results_dir, 'plots', 'algorithm_comparison.png')
        )

    # Convergence analysis
    histories = {}
    for name, data in results.items():
        if 'round_metrics' in data:
            histories[name] = {
                'round_loss': [r.get('train_loss') for r in data['round_metrics']],
                'client_divergence': [r.get('client_divergence') for r in data['round_metrics']
                                     if r.get('client_divergence') is not None]
            }

    if histories:
        plot_convergence_analysis(
            histories,
            title='Convergence Analysis',
            save_path=os.path.join(results_dir, 'plots', 'convergence_analysis.png')
        )

    # Generate HTML report
    create_experiment_report(
        results_dir,
        os.path.join(results_dir, 'experiment_report.html')
    )

    print(f"Analysis saved to {results_dir}/plots/")


def main():
    parser = argparse.ArgumentParser(description='Run Federated LLM Experiments')

    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'main', 'failure', 'compare', 'analyze'],
                       help='Experiment mode')
    parser.add_argument('--model', type=str, default='distilgpt2')
    parser.add_argument('--dataset', type=str, default='ag_news')
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Create base config
    base_config = ExperimentConfig(
        name='base',
        model=ModelConfig(name=args.model),
        data=DataConfig(
            dataset=args.dataset,
            num_clients=args.num_clients,
            partition_strategy='topic_skew'
        ),
        federated=FederatedConfig(
            num_rounds=args.num_rounds
        ),
        results_dir=args.results_dir,
        seed=args.seed,
        device=args.device
    )

    set_seed(args.seed)

    if args.mode == 'quick':
        print("Running quick test experiments...")
        run_main_experiments(base_config, quick=True)

    elif args.mode == 'main':
        print("Running main experiment grid...")
        run_main_experiments(base_config, quick=False)

    elif args.mode == 'failure':
        print("Running failure mode experiments...")
        run_failure_mode_experiments(base_config)

    elif args.mode == 'compare':
        print("Running algorithm comparison...")
        run_algorithm_comparison(base_config)

    elif args.mode == 'analyze':
        print("Generating analysis from existing results...")
        generate_analysis(args.results_dir)

    print("\nExperiments completed!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == '__main__':
    main()
