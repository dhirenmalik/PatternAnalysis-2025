"""Evaluation script for the manual FLAN‑T5 + LoRA training pipeline.

Combines the lightweight Hugging Face dataset inspection workflow with the
batch-oriented evaluation, reporting, and artefact saving features developed
for the custom trainer. The script:

* loads a checkpoint produced by the manual training loop (LoRA adapters)
* runs batched generation on the CSV splits created by ``data_setup.py``
* computes ROUGE scores and prints sample predictions
* optionally evaluates a subset of the Hugging Face validation split for quick
  sanity checks, mirroring the earlier ``predict.py`` behaviour
* saves predictions, ROUGE metrics, and textual reports to disk
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import RadiologyDataset, load_biolaysumm
from modules import print_model_info
from predict_single import (
    load_checkpoint as load_single_checkpoint,
    predict_single as summarize_single,
    resolve_device as resolve_single_device,
)
from utils import DataParams, DeviceParams, EvalParams, HyperParams
from transformers import PreTrainedTokenizerBase

# Align runtime defaults with training script to avoid noisy warnings.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


ROUGE_METRIC = evaluate.load("rouge")


def load_model(checkpoint_path: str, device: torch.device):
    """Load a checkpoint via the single-example helper and log model stats.

    Args:
        checkpoint_path: Directory containing the trained weights.
        device: Torch device that should host the model.

    Returns:
        Tuple of (model, tokenizer) ready for inference.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    model, tokenizer = load_single_checkpoint(checkpoint_path, device)
    print_model_info(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def prepare_csv_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    data_dir: Path,
    split: str,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    prefix: str,
    device: torch.device,
    limit_samples: Optional[int] = None,
    num_workers: int = 2,
) -> Tuple[RadiologyDataset, DataLoader]:
    """Build a ``DataLoader`` from CSV splits produced by ``data_setup.py``.

    Args:
        tokenizer: Tokenizer aligned with the fine-tuned model.
        data_dir: Directory that contains ``train.csv`` / ``val.csv`` / ``test.csv``.
        split: Dataset split to evaluate (``train``, ``val``/``validation``, or ``test``).
        batch_size: Number of samples per inference batch.
        max_source_length: Maximum encoder sequence length.
        max_target_length: Maximum decoder sequence length.
        prefix: Instruction prefix prepended to the source text.
        device: Target device used to determine pin_memory behaviour.
        limit_samples: Optional cap on the number of examples evaluated.
        num_workers: Background worker count for the dataloader.

    Returns:
        Tuple containing the dataset instance and the dataloader.
    """
    split_map = {"val": "val", "validation": "val", "test": "test", "train": "train"}
    split_key = split_map.get(split.lower(), split.lower())
    csv_path = data_dir / f"{split_key}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV split not found: {csv_path}")

    dataset = RadiologyDataset(
        csv_path=str(csv_path),
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        prefix=prefix,
    )

    if limit_samples is not None:
        dataset.data = dataset.data.head(limit_samples).reset_index(drop=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return dataset, dataloader


def _resolve_raw_example(dataset, idx: int) -> Tuple[str, str, str]:
    """Return raw input text, reference summary, and prefix for a dataset index.

    Args:
        dataset: `RadiologyDataset` or `Subset` wrapping it.
        idx: Zero-based index in the loader iteration order.

    Returns:
        Tuple with (report_text, lay_summary, prefix).
    """

    if isinstance(dataset, Subset):
        base_idx = dataset.indices[idx]
        return _resolve_raw_example(dataset.dataset, base_idx)

    if hasattr(dataset, "data"):
        row = dataset.data.iloc[idx]
        prefix = getattr(dataset, "prefix", "")
        return row["report_text"], row["lay_summary"], prefix

    raise ValueError("Dataset does not expose raw text (expected RadiologyDataset).")


# ---------------------------------------------------------------------------
# Generation & evaluation
# ---------------------------------------------------------------------------

def generate_predictions(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    dataloader: DataLoader,
    device: torch.device,
    prompt_prefix: Optional[str],
    max_source_length: int,
    max_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Generate predictions for every batch in the dataloader.

    Args:
        model: Fine-tuned model used for generation.
        tokenizer: Tokenizer for decoding predictions and inputs.
        dataloader: Loader that yields batches from the evaluation split.
        device: Device on which inference runs.
        prompt_prefix: Optional override for the dataset prefix. If ``None`` the
            dataset-specified prefix is used.
        max_source_length: Truncation length for encoder inputs.
        max_length: Maximum generation length.
        num_beams: Beam search width.
        no_repeat_ngram_size: Prevents repeating n-grams during generation.

    Returns:
        Tuple of predicted summaries, reference summaries, and decoded inputs.
    """
    model.eval()
    predictions: List[str] = []
    references: List[str] = []
    decoded_inputs: List[str] = []

    dataset = dataloader.dataset
    position = 0

    for batch in tqdm(dataloader, desc="Generating"):
        batch_size = len(batch["input_ids"])

        for offset in range(batch_size):
            raw_text, reference_text, dataset_prefix = _resolve_raw_example(dataset, position + offset)
            prefix_to_use = prompt_prefix if prompt_prefix is not None else dataset_prefix
            prompt = prefix_to_use + raw_text

            summary = summarize_single(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_input_len=max_source_length,
                max_target_len=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            predictions.append(summary)
            references.append(reference_text)
            decoded_inputs.append(raw_text)

        position += batch_size

    return predictions, references, decoded_inputs


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE metrics for a list of predictions and references.

    Args:
        predictions: Generated summaries.
        references: Ground-truth summaries aligned with ``predictions``.

    Returns:
        Dictionary of ROUGE-1/2/L/Lsum scores.
    """
    formatted_preds = ["\n".join(pred.strip().split(".")) or "empty" for pred in predictions]
    formatted_refs = ["\n".join(ref.strip().split(".")) or "empty" for ref in references]

    result = ROUGE_METRIC.compute(
        predictions=formatted_preds,
        references=formatted_refs,
        use_stemmer=True,
        use_aggregator=True,
    )
    return {metric: float(result[metric]) for metric in ("rouge1", "rouge2", "rougeL", "rougeLsum")}


def evaluate_hf_subset(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    add_prefix: str,
    max_input_len: int,
    max_target_len: int,
    num_samples: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    split: str = "validation",
) -> Dict[str, float]:
    """Run ROUGE computation on a subset of the Hugging Face dataset.

    Mimics the original ``predict.py`` quick-evaluation flow.

    Args:
        model: Fine-tuned model for generation.
        tokenizer: Tokenizer aligned with the model.
        device: Device on which inference runs.
        add_prefix: Prompt prefix prepended to each report.
        max_input_len: Maximum encoder length.
        max_target_len: Maximum decoder length.
        num_samples: Number of validation samples to evaluate.
        num_beams: Beam search width for generation.
        no_repeat_ngram_size: Prevents repeating n-grams during generation.
        split: Dataset split to evaluate (defaults to ``validation``).

    Returns:
        ROUGE metric dictionary.
    """
    if num_samples <= 0:
        return {}

    raw = load_biolaysumm()
    dataset = raw[split]
    cols = list(dataset.features.keys())
    input_col = next((c for c in cols if "report" in c.lower() or "source" in c.lower()), cols[0])
    target_col = next((c for c in cols if "summary" in c.lower() or "target" in c.lower()), cols[-1])

    subset = dataset.select(range(min(num_samples, len(dataset))))
    predictions: List[str] = []
    references: List[str] = []

    for example in subset:
        prompt = add_prefix + example[input_col]
        summary = summarize_single(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_input_len=max_input_len,
            max_target_len=max_target_len,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        predictions.append(summary)
        references.append(example[target_col])

    rouge_scores = compute_rouge_scores(predictions, references)
    print("\n📊 Hugging Face validation subset ROUGE:")
    for key, value in rouge_scores.items():
        print(f"  {key}: {value:.4f}")
    return rouge_scores


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_examples(
    predictions: List[str],
    references: List[str],
    inputs: List[str],
    num_examples: int,
) -> None:
    """Print representative prediction examples to stdout.

    Args:
        predictions: Generated summaries to display.
        references: Ground-truth summaries aligned with ``predictions``.
        inputs: Source radiology reports corresponding to each prediction.
        num_examples: Maximum number of examples to display.

    Returns:
        None. Examples are printed to stdout.
    """
    if not predictions:
        print("\n⚠️  No predictions available to display.")
        return

    num_examples = min(num_examples, len(predictions))
    step = max(len(predictions) // num_examples, 1)

    print("\n" + "=" * 60)
    print("📝 Example Predictions")
    print("=" * 60)

    for idx in range(num_examples):
        sample_idx = min(idx * step, len(predictions) - 1)
        print(f"\nExample {idx + 1} (index {sample_idx})")
        print("-" * 60)
        print(f"\n📄 INPUT:\n{inputs[sample_idx][:400]}{'...' if len(inputs[sample_idx]) > 400 else ''}")
        print(f"\n✅ REFERENCE:\n{references[sample_idx]}")
        print(f"\n🔮 PREDICTION:\n{predictions[sample_idx]}")
        print("\n" + "-" * 60)


def save_results(
    predictions: List[str],
    references: List[str],
    inputs: List[str],
    rouge_scores: Dict[str, float],
    output_dir: Path,
) -> None:
    """Persist predictions, ROUGE scores, and summary reports to disk.

    Args:
        predictions: Generated summaries.
        references: Reference summaries aligned with ``predictions``.
        inputs: Original radiology report texts.
        rouge_scores: ROUGE metric results keyed by metric name.
        output_dir: Directory where artefacts are written.

    Returns:
        None. Generates artefact files on disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "input_report": inputs,
            "reference_summary": references,
            "predicted_summary": predictions,
        }
    )
    df.to_csv(output_dir / "predictions.csv", index=False)

    with open(output_dir / "rouge_scores.json", "w", encoding="utf-8") as fh:
        json.dump(rouge_scores, fh, indent=2)

    lengths_pred = [len(pred.split()) for pred in predictions]
    lengths_ref = [len(ref.split()) for ref in references]
    avg_pred_len = float(np.mean(lengths_pred)) if lengths_pred else 0.0
    avg_ref_len = float(np.mean(lengths_ref)) if lengths_ref else 0.0
    ratio = (avg_pred_len / avg_ref_len) if avg_ref_len else 0.0

    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 80 + "\nFLAN-T5 Radiology Summarisation Report\n" + "=" * 80 + "\n")
        fh.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write(f"Samples:   {len(predictions)}\n\n")
        fh.write("ROUGE Scores\n" + "-" * 80 + "\n")
        for key, value in rouge_scores.items():
            fh.write(f"{key.upper():<10}: {value:.4f}\n")
        fh.write("\nLength Statistics\n" + "-" * 80 + "\n")
        fh.write(f"Average prediction length: {avg_pred_len:.1f}\n")
        fh.write(f"Average reference length:  {avg_ref_len:.1f}\n")
        fh.write(f"Length ratio (pred/ref):   {ratio:.2f}\n")


def append_results_to_checkpoint(checkpoint_path: Path, rouge_scores: Dict[str, float]) -> None:
    """Append ROUGE scores to the checkpoint ``RESULTS.txt`` file.

    Args:
        checkpoint_path: Directory that owns ``RESULTS.txt``.
        rouge_scores: ROUGE metric values to append.

    Returns:
        None. Appends text to ``RESULTS.txt`` if scores are provided.
    """
    if not rouge_scores:
        return

    results_file = checkpoint_path / "RESULTS.txt"
    with open(results_file, "a", encoding="utf-8") as fh:
        fh.write("\n\nROUGE evaluation:\n")
        for key, value in rouge_scores.items():
            fh.write(f"{key}: {value:.4f}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line options for batched evaluation.

    Returns:
        Parsed command-line arguments.
    """
    eval_defaults = EvalParams()
    data_defaults = DataParams()
    device_defaults = DeviceParams()
    hp_defaults = HyperParams()

    parser = argparse.ArgumentParser(
        description="Evaluate a FLAN-T5 LoRA checkpoint on radiology summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", default="outputs_flan_t5_lora/best_model", help="Path to checkpoint directory."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=device_defaults.preferred_device,
        help="Inference device selection.",
    )
    parser.add_argument(
        "--data_dir",
        default=data_defaults.data_dir,
        help="Directory containing train/val/test CSV files.",
    )
    parser.add_argument("--split", default="val", help="Dataset split to evaluate (train/val/test).")
    parser.add_argument(
        "--batch_size", type=int, default=eval_defaults.batch_size, help="Batch size for generation."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=hp_defaults.max_input_len,
        help="Maximum encoder sequence length.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=hp_defaults.max_target_len,
        help="Maximum decoder sequence length.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=hp_defaults.max_target_len,
        help="Maximum length for generation outputs.",
    )
    parser.add_argument(
        "--num_beams", type=int, default=eval_defaults.num_beams, help="Beam search width."
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=3, help="No-repeat n-gram constraint."
    )
    parser.add_argument(
        "--num_examples", type=int, default=eval_defaults.num_examples, help="Number of examples to print."
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on samples evaluated from CSV.")
    parser.add_argument(
        "--prefix",
        default=data_defaults.prefix,
        help="Instruction prefix for inputs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store predictions; defaults to <checkpoint>/evaluation.",
    )
    parser.add_argument(
        "--hf_rouge_samples",
        type=int,
        default=eval_defaults.hf_rouge_samples,
        help="If > 0, also compute ROUGE on this many Hugging Face validation samples.",
    )
    parser.add_argument("--hf_split", default="validation", help="Hugging Face split used for optional evaluation.")
    parser.add_argument(
        "--hf_prefix",
        default=data_defaults.hf_eval_prefix,
        help="Prefix used for Hugging Face evaluation.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for batched evaluation and reporting.

    Returns:
        None. This function orchestrates I/O side effects.
    """
    args = parse_args()

    device = resolve_single_device(args.device)
    print(f"🖥️  Using {device.type.upper()} for inference.")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path / "evaluation"

    model, tokenizer = load_model(checkpoint_path=str(checkpoint_path), device=device)
    dataset, dataloader = prepare_csv_dataloader(
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        split=args.split,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        prefix=args.prefix,
        device=device,
        limit_samples=args.limit,
    )

    start = datetime.now()
    predictions, references, inputs = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        prompt_prefix=args.prefix,
        max_source_length=args.max_source_length,
        max_length=args.max_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\n⚡ Processed {len(predictions)} samples in {elapsed:.2f}s "
          f"({len(predictions) / elapsed if elapsed else 0:.2f} summaries/sec)")

    rouge_scores = compute_rouge_scores(predictions, references)
    print("\n✅ ROUGE Results:")
    for key, value in rouge_scores.items():
        print(f"  {key}: {value:.4f}")

    print_examples(predictions, references, inputs, args.num_examples)
    save_results(predictions, references, inputs, rouge_scores, output_dir)
    append_results_to_checkpoint(checkpoint_path, rouge_scores)

    if args.hf_rouge_samples > 0:
        evaluate_hf_subset(
            model=model,
            tokenizer=tokenizer,
            device=device,
            add_prefix=args.hf_prefix,
            max_input_len=args.max_source_length,
            max_target_len=args.max_target_length,
            num_samples=args.hf_rouge_samples,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            split=args.hf_split,
        )

    print("\n📁 Artefacts saved to:", output_dir.resolve())


if __name__ == "__main__":
    main()
