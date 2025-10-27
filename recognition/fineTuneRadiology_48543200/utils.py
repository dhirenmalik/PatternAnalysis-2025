"""Centralised configuration defaults and helpers for the FLAN-T5 workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HyperParams:
    """Training hyperparameters with sensible defaults."""

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_input_len: int = 1024
    max_target_len: int = 256
    logging_steps: int = 50
    eval_steps: int = 0
    save_steps: int = 0
    eval_epoch: bool = True
    save_epoch: bool = True
    subset: Optional[int] = None
    mixed_precision: bool = True
    warmup_ratio: float = 0.03
    scheduler_type: Optional[str] = "cosine"
    scheduler_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class LoRAParams:
    """Configuration for LoRA fine-tuning adapters."""

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q", "v")


@dataclass
class DataParams:
    """Paths and limits for dataset handling."""

    data_dir: str = "recognition/fineTuneRadiology_48543200/data"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    prefix: str = "summarize for a layperson: "
    hf_dataset_name: str = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
    hf_eval_prefix: str = "summarize radiology: "


@dataclass
class EvalParams:
    """Parameters controlling evaluation behaviour."""

    num_beams: int = 4
    hf_rouge_samples: int = 0
    batch_size: int = 8
    num_examples: int = 5


@dataclass
class DeviceParams:
    """Device configuration preferences."""

    preferred_device: str = "auto"  # Options: auto/cuda/mps/cpu
    use_mixed_precision: bool = True
