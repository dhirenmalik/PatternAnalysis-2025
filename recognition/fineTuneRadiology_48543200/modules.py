"""Model utilities for the custom FLAN-T5 + LoRA training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class ModelConfig:
    """Configuration bundle for FLAN-T5 + LoRA fine-tuning.

    Attributes:
        model_name: Base Hugging Face model identifier.
        use_lora: Whether to attach LoRA adapters.
        lora_r: LoRA rank (controls adapter width).
        lora_alpha: Scaling factor applied to LoRA updates.
        lora_dropout: Dropout applied inside adapters.
        target_modules: Attention projections to wrap with LoRA.
    """

    model_name: str = "google/flan-t5-base"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Tuple[str, ...] = ("q", "v")


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str = "google/flan-t5-base") -> PreTrainedTokenizerBase:
    """Load the FLAN-T5 tokenizer with basic diagnostics.

    Args:
        model_name: Hugging Face model identifier to initialise the tokenizer.

    Returns:
        The tokenizer loaded from ``model_name``.
    """
    print(f"🔤 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"   Vocab size: {tokenizer.vocab_size:,} | pad_token_id: {tokenizer.pad_token_id}")
    return tokenizer


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    model_name: str = "google/flan-t5-base",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[Tuple[str, ...]] = None,
    move_to_device: bool = False,
) -> AutoModelForSeq2SeqLM:
    """Construct a FLAN-T5 model optionally equipped with LoRA adapters.

    Args:
        model_name: Backbone model checkpoint.
        use_lora: Enable LoRA parameter-efficient fine-tuning.
        lora_r: Adapter rank.
        lora_alpha: Adapter scaling.
        lora_dropout: Adapter dropout rate.
        target_modules: Tuple of attention projections to adapt.
        move_to_device: Move model to GPU immediately if available.

    Returns:
        The loaded FLAN-T5 model, optionally wrapped with LoRA adapters.
    """
    print("=" * 60)
    print(f"🧠 Loading base model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Base parameters: {total_params:,}")

    if use_lora:
        target_modules = target_modules or ("q", "v")
        print("🔧 Applying LoRA adapters:")
        print(f"   r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules}")

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=list(target_modules),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = 100 * trainable_params / total_params
        print(f"   Trainable params with LoRA: {trainable_params:,} ({pct:.2f}%)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Detected device: {device}")

    if move_to_device and device == "cuda":
        model = model.to(device)
        print("🚀 Model moved to GPU for manual training loop.")
    else:
        print("📍 Model left on CPU — training script will handle device placement.")

    return model


# ---------------------------------------------------------------------------
# Unified loader used across training scripts
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: Optional[ModelConfig] = None,
    move_to_device: Optional[bool] = None,
) -> Tuple[AutoModelForSeq2SeqLM, PreTrainedTokenizerBase, ModelConfig]:
    """Load both model and tokenizer using a shared configuration.

    Args:
        cfg: Optional ``ModelConfig``. If omitted the defaults are used.
        move_to_device: Override for immediate device transfer.

    Returns:
        Tuple of (model, tokenizer, resolved configuration).
    """
    cfg = cfg or ModelConfig()
    move_flag = False if move_to_device is None else move_to_device

    print("=" * 60)
    print("📦 Loading FLAN-T5 model + tokenizer")
    print("=" * 60)

    tokenizer = load_tokenizer(cfg.model_name)
    model = build_model(
        model_name=cfg.model_name,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        move_to_device=move_flag,
    )

    print("✅ Model and tokenizer ready.")
    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Utility helpers for manual inspection
# ---------------------------------------------------------------------------

def print_model_info(model: torch.nn.Module) -> None:
    """Display total, trainable, and frozen parameter counts.

    Args:
        model: Model whose parameters should be summarised.

    Returns:
        None. Details are printed to stdout.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    pct = 100 * trainable_params / total_params

    print("=" * 60)
    print("📊 Model Parameter Summary")
    print("=" * 60)
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters:    {frozen_params:,}")
    print(f"   Trainable percentage: {pct:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    # Basic smoke test for manual execution.
    config = ModelConfig()
    model, tokenizer, _ = load_model_and_tokenizer(config, move_to_device=False)
    print_model_info(model)
