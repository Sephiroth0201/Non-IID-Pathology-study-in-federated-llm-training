# Failure Modes of Federated Fine-Tuning of Language Models Under Extreme Non-IID Data Distributions

A comprehensive study of how federated learning algorithms fail when training large language models under extreme data heterogeneity.

## Research Questions

1. When does federated training of LLMs **break down** under data heterogeneity?
2. What kinds of non-IID splits hurt convergence, generalization, and representation quality?
3. How do FedAvg, FedProx, and SCAFFOLD **fail differently**?
4. Can data-aware client clustering fix these failures?

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── partitioner.py      # Non-IID partitioning strategies
│   │   └── datasets.py         # Dataset loading and preprocessing
│   ├── models/
│   │   └── lora_model.py       # LoRA fine-tuning for LLMs
│   ├── federated/
│   │   ├── client.py           # FL client implementations
│   │   ├── server.py           # FL server implementations
│   │   ├── fedavg.py           # FedAvg algorithm
│   │   ├── fedprox.py          # FedProx algorithm
│   │   └── scaffold.py         # SCAFFOLD algorithm
│   ├── metrics/
│   │   └── metrics.py          # Metrics collection and failure detection
│   └── utils/
│       ├── config.py           # Configuration management
│       ├── logging_utils.py    # Logging utilities
│       └── visualization.py    # Plotting and analysis
├── configs/
│   └── default.yaml            # Default experiment configuration
├── train.py                    # Main training script
├── run_experiments.py          # Systematic experiment runner
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Experiment
```bash
# Run FedAvg with topic skew partition
python train.py --algorithm fedavg --partition topic_skew --num-rounds 50

# Run FedProx with style skew
python train.py --algorithm fedprox --partition style_skew --mu 0.01

# Run SCAFFOLD with token skew
python train.py --algorithm scaffold --partition token_skew
```

### Using Configuration File
```bash
python train.py --config configs/default.yaml
```

### Systematic Experiments
```bash
# Quick test (fewer rounds)
python run_experiments.py --mode quick

# Full experiment grid
python run_experiments.py --mode main

# Failure mode analysis
python run_experiments.py --mode failure

# Generate analysis from results
python run_experiments.py --mode analyze
```

## Non-IID Partition Types

### 1. Topic Skew (`topic_skew`)
Each client receives data from specific topics/domains using Dirichlet distribution.
- Parameter: `alpha` (lower = more skewed)
- Tests: vocabulary and concept mismatch

### 2. Style Skew (`style_skew`)
Clients differ in writing style characteristics.
- Criteria: `length`, `formality`, `complexity`
- Tests: syntactic and stylistic drift

### 3. Token Distribution Skew (`token_skew`)
Artificially manipulated token frequencies.
- Types: `frequency`, `truncation`, `vocabulary`
- Tests: gradient bias across clients

## Federated Learning Algorithms

| Algorithm | Description | Key Parameter |
|-----------|-------------|---------------|
| **FedAvg** | Baseline weighted averaging | - |
| **FedProx** | Proximal term for drift control | `mu` (regularization) |
| **SCAFFOLD** | Control variates for variance reduction | - |

## Metrics Tracked

### Language Quality
- Perplexity
- Validation loss

### Training Stability
- Gradient norm variance
- Client update divergence
- Weight cosine drift

### Failure Detection
- Divergence detection
- Oscillation detection
- Client drift monitoring
- Convergence speed analysis

## Configuration Options

```yaml
model:
  name: distilgpt2          # distilgpt2, gpt2, tinyllama, qwen-0.5b
  lora_r: 8                 # LoRA rank
  lora_alpha: 16            # LoRA alpha
  max_length: 128           # Sequence length

data:
  dataset: ag_news          # ag_news, wikitext
  partition_strategy: topic_skew
  num_clients: 10
  alpha: 0.1                # Dirichlet concentration

federated:
  algorithm: fedavg         # fedavg, fedprox, scaffold
  num_rounds: 50
  participation_rate: 1.0
  local_epochs: 1
  learning_rate: 5e-5
  mu: 0.01                  # FedProx only
```

## Expected Results

The experiments will reveal:
1. **IID baseline**: Stable convergence across all algorithms
2. **Topic skew**: FedAvg diverges, FedProx/SCAFFOLD handle better
3. **Style skew**: Syntactic drift causes gradient variance
4. **Token skew**: Vocabulary bias affects all algorithms
5. **Low participation**: Oscillation and slow convergence
6. **High learning rate**: Divergence especially under non-IID

## References

- **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020
- **SCAFFOLD**: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", ICML 2020
