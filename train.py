#!/usr/bin/env python3
"""
Main Training Script for Federated LLM Experiments

Usage:
    python train.py --config configs/default.yaml
    python train.py --algorithm fedavg --dataset ag_news --partition topic_skew
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.partitioner import NonIIDPartitioner
from src.data.datasets import (
    prepare_federated_data,
    create_raw_dataset_for_partitioning,
    SimpleDataset
)
from src.models.lora_model import get_model_and_tokenizer, create_lora_model, get_device
from src.federated.fedavg import FedAvg
from src.federated.fedprox import FedProx
from src.federated.scaffold import SCAFFOLD
from src.metrics.metrics import MetricsCollector, FailureModeDetector
from src.utils.config import (
    ExperimentConfig, ModelConfig, DataConfig, FederatedConfig,
    load_config, save_config
)
from src.utils.logging_utils import ExperimentLogger


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_from_config(config: ExperimentConfig) -> torch.device:
    """Get torch device from config."""
    if config.device == 'auto':
        return get_device()
    return torch.device(config.device)


def create_partitioner(config: DataConfig):
    """Create data partitioner from config."""
    kwargs = {'seed': config.seed}

    if config.partition_strategy == 'topic_skew':
        kwargs['alpha'] = config.alpha
        kwargs['label_column'] = 'label'
    elif config.partition_strategy == 'style_skew':
        kwargs['text_column'] = 'text'
        kwargs['style_criteria'] = config.style_criteria
    elif config.partition_strategy == 'token_skew':
        kwargs['text_column'] = 'text'
        kwargs['skew_type'] = config.skew_type

    return NonIIDPartitioner.create(
        strategy=config.partition_strategy,
        num_clients=config.num_clients,
        **kwargs
    )


def run_experiment(config: ExperimentConfig):
    """Run a single experiment with given configuration."""
    # Setup
    set_seed(config.seed)
    device = get_device_from_config(config)
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = ExperimentLogger(
        experiment_name=config.name,
        log_dir=os.path.join(config.results_dir, 'logs'),
        use_wandb=False,
        use_tensorboard=False
    )
    logger.log_config({
        'model': config.model.__dict__,
        'data': config.data.__dict__,
        'federated': config.federated.__dict__
    })

    # Load model and tokenizer
    print(f"Loading model: {config.model.name}")
    base_model, tokenizer = get_model_and_tokenizer(
        model_name=config.model.name,
        use_quantization=config.model.use_quantization
    )

    # Apply LoRA
    model = create_lora_model(
        base_model,
        model_name=config.model.name,
        lora_r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        use_quantization=config.model.use_quantization
    )
    model.to(device)

    # Create partitioner
    print(f"Creating {config.data.partition_strategy} partitioner for {config.data.num_clients} clients")
    partitioner = create_partitioner(config.data)

    # Load raw data for partitioning
    max_samples = getattr(config.data, 'max_samples', 0)
    if max_samples > 0:
        print(f"Limiting to {max_samples} samples for quick testing")
    raw_data = create_raw_dataset_for_partitioning(config.data.dataset, 'train', max_samples)
    raw_dataset = SimpleDataset(raw_data)
    print(f"Total samples: {len(raw_dataset)}")

    # Partition data
    partition = partitioner.partition(raw_dataset)
    partition_stats = partitioner.get_statistics(raw_dataset, partition)
    print(f"Partition statistics: {partition_stats['samples_per_client']}")

    # Prepare federated data
    print("Preparing federated dataset...")
    federated_data, test_loader = prepare_federated_data(
        dataset_name=config.data.dataset,
        tokenizer=tokenizer,
        partitioner=partitioner,
        batch_size=config.data.batch_size,
        max_length=config.model.max_length
    )

    # Create client dataloaders
    client_dataloaders = {}
    for client_id in federated_data.get_all_client_ids():
        client_dataloaders[client_id] = federated_data.get_client_dataloader(client_id)

    # Initialize algorithm
    print(f"Initializing {config.federated.algorithm.upper()} algorithm")
    algorithm_class = {
        'fedavg': FedAvg,
        'fedprox': FedProx,
        'scaffold': SCAFFOLD
    }[config.federated.algorithm]

    algo_kwargs = {
        'model': model,
        'num_clients': config.data.num_clients,
        'client_dataloaders': client_dataloaders,
        'test_dataloader': test_loader,
        'device': device,
        'participation_rate': config.federated.participation_rate,
        'local_epochs': config.federated.local_epochs,
        'learning_rate': config.federated.learning_rate,
        'num_rounds': config.federated.num_rounds,
        'seed': config.seed
    }

    if config.federated.algorithm == 'fedprox':
        algo_kwargs['mu'] = config.federated.mu

    trainer = algorithm_class(**algo_kwargs)

    # Setup metrics collection
    metrics_collector = MetricsCollector(log_dir=config.results_dir)
    failure_detector = FailureModeDetector()

    def training_callback(round_num: int, stats: dict):
        """Callback for logging during training."""
        metrics_collector.log_round(
            round_num=round_num,
            train_loss=stats['train_loss'],
            client_divergence=stats.get('divergence'),
            gradient_variance=stats.get('gradient_variance')
        )
        failure_detector.update(
            stats['train_loss'],
            stats.get('divergence', 0)
        )

        # Log to experiment logger
        logger.log_round(round_num, {
            'train_loss': stats['train_loss'],
            'divergence': stats.get('divergence', 0),
            'gradient_variance': stats.get('gradient_variance', 0)
        })

    # Run training
    print(f"\nStarting training for {config.federated.num_rounds} rounds...")
    print("=" * 60)

    history = trainer.train(
        eval_every=config.eval_every,
        verbose=True,
        callback=training_callback
    )

    # Final evaluation
    test_loss, perplexity = trainer.evaluate()
    print("=" * 60)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Perplexity: {perplexity:.2f}")

    # Check for failure modes
    failure_status = failure_detector.get_status()
    recommendations = failure_detector.get_recommendations()
    print("\nFailure Mode Analysis:")
    for key, value in failure_status.items():
        print(f"  {key}: {value}")
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")

    # Save results
    summary = metrics_collector.get_summary()
    summary['final_test_loss'] = test_loss
    summary['final_perplexity'] = perplexity
    summary['failure_status'] = failure_status

    logger.log_summary(summary)

    # Save metrics
    metrics_path = metrics_collector.save(f'{config.name}_metrics.json')
    print(f"\nMetrics saved to: {metrics_path}")

    # Save checkpoint
    if config.save_checkpoints:
        checkpoint_path = os.path.join(
            config.checkpoint_dir,
            f'{config.name}_final.pt'
        )
        trainer.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

    # Save config
    config_path = os.path.join(config.results_dir, f'{config.name}_config.yaml')
    save_config(config, config_path)

    logger.close()

    return history, summary


def main():
    parser = argparse.ArgumentParser(
        description='Federated LLM Training - Non-IID Pathology Study'
    )

    # Config file
    parser.add_argument('--config', type=str, help='Path to config file')

    # Override options
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--model', type=str, default='distilgpt2',
                       choices=['distilgpt2', 'gpt2', 'tinyllama', 'qwen-0.5b'])
    parser.add_argument('--dataset', type=str, default='ag_news',
                       choices=['wikitext', 'ag_news'])
    parser.add_argument('--algorithm', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'scaffold'])
    parser.add_argument('--partition', type=str, default='topic_skew',
                       choices=['iid', 'topic_skew', 'style_skew', 'token_skew'])
    parser.add_argument('--num-clients', type=int, default=10)
    parser.add_argument('--num-rounds', type=int, default=50)
    parser.add_argument('--participation-rate', type=float, default=1.0)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx mu')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Dirichlet alpha for topic_skew')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0,
                       help='Max training samples (0=all, use 1000-5000 for quick tests)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--results-dir', type=str, default='results')

    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = load_config(args.config)
        # Apply overrides
        if args.name:
            config.name = args.name
    else:
        # Create config from arguments
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = args.name or f"{args.algorithm}_{args.partition}_{timestamp}"

        config = ExperimentConfig(
            name=name,
            model=ModelConfig(
                name=args.model,
                max_length=args.max_length
            ),
            data=DataConfig(
                dataset=args.dataset,
                partition_strategy=args.partition,
                num_clients=args.num_clients,
                alpha=args.alpha,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                seed=args.seed
            ),
            federated=FederatedConfig(
                algorithm=args.algorithm,
                num_rounds=args.num_rounds,
                participation_rate=args.participation_rate,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                mu=args.mu
            ),
            seed=args.seed,
            device=args.device,
            results_dir=args.results_dir
        )

    print(f"Running experiment: {config.name}")
    print(f"Algorithm: {config.federated.algorithm}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Partition: {config.data.partition_strategy}")
    print(f"Clients: {config.data.num_clients}")
    print(f"Rounds: {config.federated.num_rounds}")
    print()

    run_experiment(config)


if __name__ == '__main__':
    main()
