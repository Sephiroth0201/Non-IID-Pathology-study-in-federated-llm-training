"""
Configuration Management for Federated Learning Experiments

Provides structured configuration for:
- Model settings
- Data partitioning
- FL algorithm parameters
- Training hyperparameters
"""

import json
import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = 'distilgpt2'  # distilgpt2, gpt2, tinyllama, qwen-0.5b
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_quantization: bool = False
    max_length: int = 128


@dataclass
class DataConfig:
    """Data and partitioning configuration."""
    dataset: str = 'ag_news'  # wikitext, ag_news
    partition_strategy: str = 'topic_skew'  # iid, topic_skew, style_skew, token_skew
    num_clients: int = 10
    alpha: float = 0.1  # Dirichlet concentration (for topic_skew)
    style_criteria: str = 'length'  # For style_skew: length, formality, complexity
    skew_type: str = 'frequency'  # For token_skew: frequency, truncation, vocabulary
    batch_size: int = 8
    max_samples: int = 0  # 0 = all samples, >0 = limit for quick testing
    seed: int = 42


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    algorithm: str = 'fedavg'  # fedavg, fedprox, scaffold
    num_rounds: int = 50
    participation_rate: float = 1.0
    local_epochs: int = 1
    learning_rate: float = 5e-5
    mu: float = 0.01  # FedProx proximal coefficient


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str = 'default_experiment'
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    eval_every: int = 5
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    seed: int = 42
    device: str = 'auto'  # auto, cpu, cuda, mps

    def __post_init__(self):
        # Convert dicts to dataclasses if needed
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        if isinstance(self.federated, dict):
            self.federated = FederatedConfig(**self.federated)


def load_config(path: str) -> ExperimentConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        ExperimentConfig instance
    """
    with open(path, 'r') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return ExperimentConfig(**data)


def save_config(config: ExperimentConfig, path: str):
    """
    Save configuration to YAML or JSON file.

    Args:
        config: ExperimentConfig instance
        path: Output path
    """
    data = asdict(config)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    with open(path, 'w') as f:
        if path.endswith('.yaml') or path.endswith('.yml'):
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2)


def create_experiment_configs(base_config: ExperimentConfig) -> List[ExperimentConfig]:
    """
    Generate experiment configurations for systematic studies.

    Creates configs for:
    - All algorithms (FedAvg, FedProx, SCAFFOLD)
    - All non-IID types (IID, topic, style, token skew)
    - Various participation rates
    """
    configs = []

    algorithms = ['fedavg', 'fedprox', 'scaffold']
    partitions = ['iid', 'topic_skew', 'style_skew', 'token_skew']
    participation_rates = [0.3, 0.5, 1.0]

    for algo in algorithms:
        for partition in partitions:
            for rate in participation_rates:
                config = ExperimentConfig(
                    name=f"{algo}_{partition}_pr{int(rate*100)}",
                    model=ModelConfig(**asdict(base_config.model)),
                    data=DataConfig(
                        **{**asdict(base_config.data), 'partition_strategy': partition}
                    ),
                    federated=FederatedConfig(
                        **{**asdict(base_config.federated),
                           'algorithm': algo,
                           'participation_rate': rate}
                    ),
                    eval_every=base_config.eval_every,
                    save_checkpoints=base_config.save_checkpoints,
                    checkpoint_dir=base_config.checkpoint_dir,
                    results_dir=base_config.results_dir,
                    seed=base_config.seed,
                    device=base_config.device
                )
                configs.append(config)

    return configs


def create_failure_mode_configs(base_config: ExperimentConfig) -> List[ExperimentConfig]:
    """
    Generate configurations for failure mode experiments.

    Tests:
    - Extreme topic isolation
    - Very low participation
    - High learning rate
    """
    configs = []

    # Extreme topic isolation (very low alpha)
    config = ExperimentConfig(
        name="extreme_topic_isolation",
        model=ModelConfig(**asdict(base_config.model)),
        data=DataConfig(
            **{**asdict(base_config.data),
               'partition_strategy': 'topic_skew',
               'alpha': 0.01}  # Very concentrated
        ),
        federated=FederatedConfig(**asdict(base_config.federated)),
        eval_every=base_config.eval_every,
        seed=base_config.seed,
        device=base_config.device
    )
    configs.append(config)

    # Very low participation
    config = ExperimentConfig(
        name="low_participation",
        model=ModelConfig(**asdict(base_config.model)),
        data=DataConfig(**asdict(base_config.data)),
        federated=FederatedConfig(
            **{**asdict(base_config.federated),
               'participation_rate': 0.1}  # Only 10% participation
        ),
        eval_every=base_config.eval_every,
        seed=base_config.seed,
        device=base_config.device
    )
    configs.append(config)

    # High learning rate
    config = ExperimentConfig(
        name="high_learning_rate",
        model=ModelConfig(**asdict(base_config.model)),
        data=DataConfig(**asdict(base_config.data)),
        federated=FederatedConfig(
            **{**asdict(base_config.federated),
               'learning_rate': 1e-3}  # 20x higher
        ),
        eval_every=base_config.eval_every,
        seed=base_config.seed,
        device=base_config.device
    )
    configs.append(config)

    # Many local epochs (more drift)
    config = ExperimentConfig(
        name="many_local_epochs",
        model=ModelConfig(**asdict(base_config.model)),
        data=DataConfig(**asdict(base_config.data)),
        federated=FederatedConfig(
            **{**asdict(base_config.federated),
               'local_epochs': 5}
        ),
        eval_every=base_config.eval_every,
        seed=base_config.seed,
        device=base_config.device
    )
    configs.append(config)

    return configs
