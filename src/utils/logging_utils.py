"""
Logging Utilities for Federated Learning Experiments
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'federated_llm',
    log_dir: str = 'logs',
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Enable console output
        file: Enable file output

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """
    Specialized logger for experiment tracking.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs',
        use_wandb: bool = False,
        use_tensorboard: bool = False
    ):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Setup base logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=log_dir
        )

        # Optional: wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project='federated-llm',
                    name=experiment_name,
                    reinit=True
                )
            except ImportError:
                self.logger.warning("wandb not installed, skipping")

        # Optional: tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, 'tensorboard', experiment_name)
                self.tb_writer = SummaryWriter(tb_dir)
            except ImportError:
                self.logger.warning("tensorboard not installed, skipping")

    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.logger.info(f"Configuration: {config}")

        if self.wandb_run:
            import wandb
            wandb.config.update(config)

    def log_round(
        self,
        round_num: int,
        metrics: dict
    ):
        """Log metrics for a training round."""
        msg = f"Round {round_num}: " + ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.logger.info(msg)

        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=round_num)

        if self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, round_num)

    def log_summary(self, summary: dict):
        """Log experiment summary."""
        self.logger.info("=" * 50)
        self.logger.info("Experiment Summary")
        self.logger.info("=" * 50)
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")

        if self.wandb_run:
            import wandb
            wandb.summary.update(summary)

    def close(self):
        """Close loggers."""
        if self.wandb_run:
            import wandb
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()
