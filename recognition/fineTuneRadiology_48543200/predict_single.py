#!/usr/bin/env python3
"""Generate a single layperson summary using a trained FLAN-T5 checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedTokenizerBase

from modules import load_tokenizer, print_model_info
from utils import DataParams, DeviceParams, EvalParams, HyperParams

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def resolve_device(choice: str) -> torch.device:
    """Return an appropriate torch device for inference.

    Args:
        choice: Device preference (`auto`, `cuda`, `mps`, or `cpu`).

    Returns:
        Torch device that should host the model and inputs.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():  # pragma: no cover - macOS only
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("⚠️  CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    if choice == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():  # pragma: no cover
            return torch.device("mps")
        print("⚠️  MPS requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[AutoModelForSeq2SeqLM, PreTrainedTokenizerBase]:
    """Load a checkpoint, merging LoRA adapters if necessary.

    Args:
        checkpoint_path: Directory that stores the trained weights or adapters.
        device: Torch device onto which the model should be moved.

    Returns:
        Tuple of the loaded model (in evaluation mode) and the tokenizer.
    """
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    try:
        config = PeftConfig.from_pretrained(checkpoint_path)
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    else:
        base_name = config.base_model_name_or_path
        tokenizer = load_tokenizer(base_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        # For single inference, merge adapters for speed.
        model = model.merge_and_unload()

    model = model.to(device).eval()
    print_model_info(model)
    return model, tokenizer


def predict_single(
    model: AutoModelForSeq2SeqLM,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str,
    max_input_len: int,
    max_target_len: int,
    num_beams: int,
) -> str:
    """Generate a summary for a single prompt using beam search.

    Args:
        model: Autoregressive seq2seq model already loaded on ``device``.
        tokenizer: Tokenizer matched to the model vocabulary.
        device: Torch device used for inference.
        prompt: Text fed into the encoder, usually prefix + report.
        max_input_len: Maximum number of tokens allowed for the input prompt.
        max_target_len: Maximum number of tokens to decode.
        num_beams: Beam search width.

    Returns:
        Decoded summary string.
    """
    inputs = tokenizer(
        prompt,
        max_length=max_input_len,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=max_target_len,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    """Construct and parse CLI arguments for single-example inference.

    Returns:
        Parsed command-line arguments.
    """
    data_defaults = DataParams()
    device_defaults = DeviceParams()
    eval_defaults = EvalParams()
    hp_defaults = HyperParams()

    parser = argparse.ArgumentParser(
        description="Generate a single lay summary from a trained FLAN-T5 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs_flan_t5_lora/best_model",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Radiology report text to summarise.",
    )
    parser.add_argument(
        "--prefix",
        default=data_defaults.prefix,
        help="Instruction prefix prepended to the input text.",
    )
    parser.add_argument("--max_input_len", type=int, default=hp_defaults.max_input_len)
    parser.add_argument("--max_target_len", type=int, default=hp_defaults.max_target_len)
    parser.add_argument("--num_beams", type=int, default=eval_defaults.num_beams)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=device_defaults.preferred_device,
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for one-off summary generation from the command line.

    Returns:
        None. Outputs are printed directly to stdout.
    """
    args = parse_args()
    device = resolve_device(args.device)
    print(f"🖥️  Using {device.type.upper()} for inference.")

    checkpoint = Path(args.checkpoint)
    model, tokenizer = load_checkpoint(checkpoint, device)

    prompt = args.prefix + args.text.strip()
    summary = predict_single(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        num_beams=args.num_beams,
    )

    print("\n=== Prediction ===")
    print(summary)


if __name__ == "__main__":
    main()
