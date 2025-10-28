# @file utils.py
# @brief Shared configuration dataclasses and utilities for the training stack.
# @author Dhiren Malik (48543200)

"""Centralised configuration defaults and helpers for the FLAN-T5 workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HyperParams:
    """Training hyperparameters with sensible defaults.

    Attributes:
        epochs: Number of passes over the training dataset.
        batch_size: Number of samples processed per optimisation step.
        learning_rate: Base learning rate fed into the optimiser.
        weight_decay: L2 penalty applied via AdamW.
        gradient_accumulation_steps: Mini-batches accumulated before each backward step.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        max_input_len: Maximum number of tokens on the encoder side.
        max_target_len: Maximum number of tokens on the decoder side.
        logging_steps: Interval for CLI logging.
        eval_steps: Step cadence for validation (0 disables).
        save_steps: Step cadence for checkpointing (0 disables).
        eval_epoch: Whether to force evaluation at epoch boundaries.
        save_epoch: Whether to force checkpointing at epoch boundaries.
        subset: Optional sample cap for quick experiments.
        mixed_precision: Enable automatic mixed precision when available.
        warmup_ratio: Fraction of total steps used for LR warmup.
        scheduler_type: Optional scheduler identifier (e.g. ``cosine``).
        scheduler_params: Extra scheduler keyword arguments.
    """

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
    """Configuration for LoRA fine-tuning adapters.

    Attributes:
        enabled: Toggle for attaching LoRA modules.
        r: Low-rank dimension used inside the adapters.
        alpha: Scaling factor applied to LoRA updates.
        dropout: Dropout probability within LoRA modules.
        target_modules: Attention projections wrapped with LoRA adapters.
    """

    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q", "v")


@dataclass
class DataParams:
    """Paths and limits for dataset handling.

    Attributes:
        data_dir: Root directory for the processed CSV splits.
        train_file: File name for the training split.
        val_file: File name for the validation split.
        test_file: File name for the hold-out split.
        prefix: Instruction prefix prepended during tokenisation.
        hf_dataset_name: Hugging Face dataset identifier for BioLaySumm.
        hf_eval_prefix: Prompt used when evaluating Hugging Face subsets.
    """

    data_dir: str = "recognition/fineTuneRadiology_48543200/data"
    train_file: str = "train.csv"
    val_file: str = "val.csv"
    test_file: str = "test.csv"
    prefix: str = "summarize for a layperson: "
    hf_dataset_name: str = "BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track"
    hf_eval_prefix: str = "summarize radiology: "


@dataclass
class EvalParams:
    """Parameters controlling evaluation behaviour.

    Attributes:
        num_beams: Beam search width for generation.
        hf_rouge_samples: Number of Hugging Face samples evaluated ad hoc.
        batch_size: Batch size for prediction.
        num_examples: Number of examples printed to stdout.
    """

    num_beams: int = 4
    hf_rouge_samples: int = 0
    batch_size: int = 8
    num_examples: int = 5


@dataclass
class DeviceParams:
    """Device configuration preferences.

    Attributes:
        preferred_device: Default runtime choice (`auto`, `cuda`, `mps`, `cpu`).
        use_mixed_precision: Enables AMP when compatible hardware is present.
    """

    preferred_device: str = "auto"  # Options: auto/cuda/mps/cpu
    use_mixed_precision: bool = True
