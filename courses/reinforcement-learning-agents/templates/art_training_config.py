"""
ART Training Configuration Template
====================================

Production-ready configuration for training agents with ART (Agent Reinforcement Trainer).
Copy this file to your project and modify the settings for your use case.

Usage:
    from art_training_config import get_config
    config = get_config(model_name="Qwen/Qwen2.5-7B", task="sql_agent")
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model and LoRA configuration."""
    model_name: str = "Qwen/Qwen2.5-3B"
    max_seq_length: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class GRPOConfig:
    """GRPO training hyperparameters."""
    group_size: int = 4          # N completions per prompt
    learning_rate: float = 1e-5
    kl_coeff: float = 0.05       # KL divergence penalty weight
    clip_epsilon: float = 0.2    # PPO-style clipping
    num_train_steps: int = 100
    batch_size: int = 4          # prompts per batch
    max_completion_length: int = 1024
    temperature: float = 0.7     # sampling temperature for rollouts


@dataclass
class RULERConfig:
    """RULER automatic reward configuration."""
    judge_model: str = "openai/o4-mini"
    scoring_mode: str = "relative"  # "relative" or "absolute"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 512


@dataclass
class InferenceConfig:
    """vLLM inference server configuration."""
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    tensor_parallel_size: int = 1  # increase for multi-GPU
    port: int = 8001


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    ruler: RULERConfig = field(default_factory=RULERConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 10

    # Logging
    log_dir: str = "./logs"
    log_every_n_steps: int = 1

    # MCP server
    mcp_server_url: Optional[str] = None
    mcp_server_port: int = 8000


# ============================================================
# Preset configurations for common tasks
# ============================================================

PRESETS = {
    "sql_agent_small": TrainingConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-3B", max_seq_length=4096),
        grpo=GRPOConfig(group_size=4, num_train_steps=50, learning_rate=2e-5),
        ruler=RULERConfig(judge_model="openai/o4-mini"),
    ),
    "sql_agent_medium": TrainingConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-7B", max_seq_length=4096),
        grpo=GRPOConfig(group_size=4, num_train_steps=100, learning_rate=1e-5),
        ruler=RULERConfig(judge_model="openai/o4-mini"),
    ),
    "sql_agent_large": TrainingConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-14B", max_seq_length=8192),
        grpo=GRPOConfig(group_size=8, num_train_steps=200, learning_rate=5e-6),
        ruler=RULERConfig(judge_model="openai/o4-mini"),
        inference=InferenceConfig(tensor_parallel_size=2),
    ),
    "tool_agent_small": TrainingConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-3B", max_seq_length=8192),
        grpo=GRPOConfig(
            group_size=4, num_train_steps=80,
            max_completion_length=2048, learning_rate=2e-5,
        ),
        ruler=RULERConfig(judge_model="openai/o4-mini"),
    ),
}


def get_config(preset: str = "sql_agent_small", **overrides) -> TrainingConfig:
    """
    Get a training configuration from a preset with optional overrides.

    Parameters
    ----------
    preset : str
        One of: sql_agent_small, sql_agent_medium, sql_agent_large, tool_agent_small
    **overrides
        Override specific fields. Use dot notation for nested fields:
        get_config("sql_agent_small", model_name="Qwen/Qwen2.5-7B")

    Returns
    -------
    TrainingConfig
    """
    if preset not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    config = PRESETS[preset]

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.grpo, key):
            setattr(config.grpo, key, value)
        elif hasattr(config.ruler, key):
            setattr(config.ruler, key, value)
        elif hasattr(config.inference, key):
            setattr(config.inference, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config field: {key}")

    return config


def estimate_training_cost(config: TrainingConfig) -> dict:
    """
    Estimate GPU hours and cost for a training run.

    Returns
    -------
    dict with estimated gpu_hours, cost_usd, and recommendations
    """
    # Rough estimates based on model size
    model_size = config.model.model_name.split("-")[-1].upper()

    size_to_hours = {
        "3B": 2.0,
        "7B": 5.0,
        "14B": 12.0,
        "32B": 30.0,
    }

    base_hours = size_to_hours.get(model_size, 5.0)

    # Scale by training steps
    step_multiplier = config.grpo.num_train_steps / 100

    # Scale by group size
    group_multiplier = config.grpo.group_size / 4

    gpu_hours = base_hours * step_multiplier * group_multiplier

    # Cost at ~$2/hr for A100 spot instance
    cost_per_hour = 2.0
    cost_usd = gpu_hours * cost_per_hour

    # GPU recommendation
    if model_size in ("3B", "7B"):
        gpu_rec = "1x A100 40GB or 1x RTX 4090 24GB"
    elif model_size == "14B":
        gpu_rec = "1x A100 80GB or 2x RTX 4090 24GB"
    else:
        gpu_rec = "2x A100 80GB"

    return {
        "estimated_gpu_hours": round(gpu_hours, 1),
        "estimated_cost_usd": round(cost_usd, 2),
        "gpu_recommendation": gpu_rec,
        "model": config.model.model_name,
        "train_steps": config.grpo.num_train_steps,
        "group_size": config.grpo.group_size,
    }


if __name__ == "__main__":
    # Print all presets with cost estimates
    for name, config in PRESETS.items():
        estimate = estimate_training_cost(config)
        print(f"\n{'='*50}")
        print(f"Preset: {name}")
        print(f"  Model:      {estimate['model']}")
        print(f"  Steps:      {estimate['train_steps']}")
        print(f"  Group size: {estimate['group_size']}")
        print(f"  GPU hours:  ~{estimate['estimated_gpu_hours']}h")
        print(f"  Est. cost:  ~${estimate['estimated_cost_usd']}")
        print(f"  GPU:        {estimate['gpu_recommendation']}")
