# modules.py
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model


@dataclass
class ModelConfig:
    model_name: str = "google/flan-t5-base"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Target modules for T5 attention. Keeping it minimal = more stable on small VRAM.
    target_modules: Tuple[str, ...] = ("q", "v")


def load_model_and_tokenizer(cfg: Optional[ModelConfig] = None):
    """
    Loads tokenizer + model. If cfg.use_lora == True, wraps the model with PEFT LoRA adapters.
    """
    cfg = cfg or ModelConfig()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    if cfg.use_lora:
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
            target_modules=list(cfg.target_modules),
        )
        model = get_peft_model(model, lora_cfg)

    # Colab convenience: put model on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, cfg
