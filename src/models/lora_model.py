"""
Model Module with LoRA Fine-Tuning Support

Supports:
- DistilGPT2 (lightweight, easy to fine-tune)
- TinyLLaMA-1.1B (quantized for Mac feasibility)
- Qwen-0.5B (strong but manageable)

Uses PEFT library for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)
from typing import Dict, List, Optional, Tuple, Any
import copy


# Supported models with their configurations
MODEL_CONFIGS = {
    'distilgpt2': {
        'model_name': 'distilgpt2',
        'target_modules': ['c_attn', 'c_proj'],
        'default_lora_r': 8,
        'default_lora_alpha': 16,
    },
    'gpt2': {
        'model_name': 'gpt2',
        'target_modules': ['c_attn', 'c_proj'],
        'default_lora_r': 8,
        'default_lora_alpha': 16,
    },
    'tinyllama': {
        'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'default_lora_r': 16,
        'default_lora_alpha': 32,
    },
    'qwen-0.5b': {
        'model_name': 'Qwen/Qwen2-0.5B',
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'default_lora_r': 8,
        'default_lora_alpha': 16,
    }
}


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_model_and_tokenizer(
    model_name: str = 'distilgpt2',
    use_quantization: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and tokenizer.

    Args:
        model_name: One of 'distilgpt2', 'gpt2', 'tinyllama', 'qwen-0.5b'
        use_quantization: Whether to use 4-bit quantization (for larger models)

    Returns:
        Tuple of (model, tokenizer)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]
    hf_model_name = config['model_name']

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config for larger models
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

    # Load model
    device = get_device()
    model_kwargs = {
        'trust_remote_code': True,
    }

    if quantization_config and device.type == 'cuda':
        model_kwargs['quantization_config'] = quantization_config
        model_kwargs['device_map'] = 'auto'
    else:
        model_kwargs['torch_dtype'] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        **model_kwargs
    )

    if not use_quantization and device.type != 'cuda':
        model = model.to(device)

    return model, tokenizer


def create_lora_model(
    model: PreTrainedModel,
    model_name: str = 'distilgpt2',
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    use_quantization: bool = False
) -> PeftModel:
    """
    Apply LoRA to a base model for parameter-efficient fine-tuning.

    Args:
        model: Base pretrained model
        model_name: Model type for target module selection
        lora_r: LoRA rank (default from MODEL_CONFIGS)
        lora_alpha: LoRA alpha scaling (default from MODEL_CONFIGS)
        lora_dropout: Dropout for LoRA layers
        use_quantization: Whether model uses quantization

    Returns:
        PeftModel with LoRA applied
    """
    config = MODEL_CONFIGS[model_name]

    r = lora_r or config['default_lora_r']
    alpha = lora_alpha or config['default_lora_alpha']

    # Prepare model for k-bit training if quantized
    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=config['target_modules'],
        bias='none',
    )

    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return peft_model


def get_lora_state_dict(model: PeftModel) -> Dict[str, torch.Tensor]:
    """
    Extract only the LoRA parameters from a PEFT model.

    This is used for federated aggregation - we only send LoRA weights.
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            state_dict[name] = param.data.clone()
    return state_dict


def set_lora_state_dict(model: PeftModel, state_dict: Dict[str, torch.Tensor]):
    """
    Load LoRA parameters into a PEFT model.

    Used after federated aggregation to update client models.
    """
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)


def average_lora_weights(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    """
    Average LoRA weights from multiple clients (FedAvg aggregation).

    Args:
        state_dicts: List of LoRA state dicts from clients
        weights: Optional weights for weighted averaging (e.g., by data size)

    Returns:
        Averaged state dict
    """
    if not state_dicts:
        raise ValueError("No state dicts provided")

    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    averaged = {}
    for key in state_dicts[0].keys():
        averaged[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )

    return averaged


def compute_weight_divergence(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute L2 divergence between two sets of LoRA weights.

    Useful for measuring client drift.
    """
    total_diff = 0.0
    total_norm = 0.0

    for key in state_dict1.keys():
        if key in state_dict2:
            diff = (state_dict1[key].float() - state_dict2[key].float()).norm()
            total_diff += diff.item() ** 2
            total_norm += state_dict1[key].float().norm().item() ** 2

    return (total_diff ** 0.5) / max(total_norm ** 0.5, 1e-8)


def compute_cosine_similarity(
    state_dict1: Dict[str, torch.Tensor],
    state_dict2: Dict[str, torch.Tensor]
) -> float:
    """
    Compute cosine similarity between flattened LoRA weight vectors.
    """
    vec1 = torch.cat([p.flatten().float() for p in state_dict1.values()])
    vec2 = torch.cat([p.flatten().float() for p in state_dict2.values()])

    cos_sim = torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    )
    return cos_sim.item()


class FederatedModel:
    """
    Wrapper for managing model in federated learning context.
    Handles LoRA weight extraction, aggregation, and updates.
    """

    def __init__(
        self,
        model_name: str = 'distilgpt2',
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_quantization: bool = False
    ):
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        self.device = get_device()

        # Load model and tokenizer
        self.base_model, self.tokenizer = get_model_and_tokenizer(
            model_name, use_quantization
        )

        # Apply LoRA
        self.model = create_lora_model(
            self.base_model,
            model_name,
            lora_r,
            lora_alpha,
            lora_dropout,
            use_quantization
        )

    def get_lora_weights(self) -> Dict[str, torch.Tensor]:
        """Get current LoRA weights."""
        return get_lora_state_dict(self.model)

    def set_lora_weights(self, state_dict: Dict[str, torch.Tensor]):
        """Set LoRA weights."""
        set_lora_state_dict(self.model, state_dict)

    def clone_for_client(self) -> 'FederatedModel':
        """Create a copy of this model for client training."""
        # Create new instance with same config
        client_model = FederatedModel(
            model_name=self.model_name,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            use_quantization=self.use_quantization
        )
        # Copy current weights
        client_model.set_lora_weights(self.get_lora_weights())
        return client_model

    def to(self, device: torch.device) -> 'FederatedModel':
        """Move model to device."""
        self.model = self.model.to(device)
        self.device = device
        return self

    def train(self):
        """Set model to training mode."""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()

    def __call__(self, **kwargs):
        """Forward pass."""
        return self.model(**kwargs)
