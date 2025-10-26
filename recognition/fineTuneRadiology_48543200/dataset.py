"""Dataset utilities for the manual BioLaySumm training workflow."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd
import torch
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

# Column name candidates seen across BioLaySumm variants.
CANDIDATE_INPUT_KEYS: Tuple[str, ...] = (
    "radiology_report",
    "expert_report",
    "report",
    "source",
)
CANDIDATE_TARGET_KEYS: Tuple[str, ...] = (
    "layman_report",
    "lay_summary",
    "summary",
    "target",
)


def _guess_column(names: Sequence[str], candidates: Sequence[str]) -> str:
    """Pick the first matching column regardless of letter casing."""
    names_lower = [col.lower() for col in names]
    for candidate in candidates:
        if candidate.lower() in names_lower:
            return names[names_lower.index(candidate.lower())]
    return names[0]


# ---------------------------------------------------------------------------
def load_biolaysumm(split_mapping: Optional[Dict[str, str]] = None) -> DatasetDict:
    """Load the BioLaySumm 2025 open-source dataset from Hugging Face.

    Args:
        split_mapping: Optional mapping from expected split names to dataset
            split identifiers. Currently unused but retained for future
            compatibility.

    Returns:
        A ``DatasetDict`` containing ``train``, ``validation`` and ``test``
        splits. Missing splits are inferred by splitting the training set.
    """
    ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    if isinstance(ds, DatasetDict) and set(ds.keys()) >= {"train", "validation", "test"}:
        return ds

    base = ds["train"] if "train" in ds else list(ds.values())[0]
    tmp = base.train_test_split(test_size=0.1, seed=42)
    val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict(train=tmp["train"], validation=val_test["train"], test=val_test["test"])


def build_tokenized(
    raw: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_input_len: int = 1024,
    max_target_len: int = 256,
    input_col: Optional[str] = None,
    target_col: Optional[str] = None,
    add_prefix: str = "summarize radiology: ",
) -> Tuple[DatasetDict, str, str]:
    """Tokenise splits using Hugging Face's mapping helpers."""
    sample = raw["train"].features
    cols = list(sample.keys())
    input_col = input_col or _guess_column(cols, CANDIDATE_INPUT_KEYS)
    target_col = target_col or _guess_column(cols, CANDIDATE_TARGET_KEYS)

    print(f"🔍 Using columns: input='{input_col}' | target='{target_col}'")

    def preprocess(batch):
        sources = [add_prefix + s for s in batch[input_col]]
        model_inputs = tokenizer(
            sources,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            batch[target_col],
            max_length=max_target_len,
            truncation=True,
            padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=cols)
    return tokenized, input_col, target_col


class RadiologyDataset(Dataset):
    """PyTorch dataset for radiology report → lay summary generation.

    Mirrors the preprocessing required by the custom training loop.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 256,
        prefix: str = "summarize for a layperson: ",
    ):
        """Initialise the dataset from a CSV file.

        Args:
            csv_path: Location of the CSV file containing ``report_text`` and
                ``lay_summary`` columns.
            tokenizer: Tokenizer used for encoding source and target text.
            max_source_length: Maximum sequence length for encoder-side text.
            max_target_length: Maximum sequence length for decoder targets.
            prefix: Task prompt prepended to each source example.
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix

        print(f"✅ Loaded {len(self.data)} samples from {Path(csv_path).name}")

    def __len__(self) -> int:
        """Return the number of rows available in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenise and return a single example.

        Args:
            idx: Zero-based row index to retrieve.

        Returns:
            A dictionary containing ``input_ids``, ``attention_mask`` and
            ``labels`` ready for model consumption.
        """
        row = self.data.iloc[idx]
        source_text = self.prefix + row["report_text"]
        target_text = row["lay_summary"]

        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_encoding["input_ids"].squeeze(),
            "attention_mask": source_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def create_dataloader(
    csv_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    shuffle: bool = True,
    max_source_length: int = 512,
    max_target_length: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Create a ``DataLoader`` compatible with the manual training loop.

    Args:
        csv_path: Path to the CSV file providing ``report_text`` / ``lay_summary``.
        tokenizer: Tokenizer used for encoding.
        batch_size: Batch size fed into the manual training loop.
        shuffle: Whether to randomise order (useful for training).
        max_source_length: Maximum encoder length in tokens.
        max_target_length: Maximum decoder length in tokens.
        num_workers: Parallel workers for background data loading.

    Returns:
        A ``DataLoader`` that yields batches compatible with the manual loop.
    """
    dataset = RadiologyDataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"📦 Created DataLoader: {len(dataloader)} batches of size {batch_size}")
    return dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

    train_dataset = RadiologyDataset(
        csv_path="recognition/fineTuneRadiology_48543200/data/train.csv",
        tokenizer=tokenizer,
    )

    print(f"\n📊 Dataset samples: {len(train_dataset)}")

    sample = train_dataset[0]
    print("\n📝 Sample tensor shapes:")
    for key, val in sample.items():
        print(f"   {key}: {val.shape} | dtype: {val.dtype}")

    print("\n🔍 Decoded glimpse:")
    input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    label_ids = sample["labels"][sample["labels"] != -100]
    target_text = tokenizer.decode(label_ids, skip_special_tokens=True)
    print(f"   Input:  {input_text[:100]}...")
    print(f"   Target: {target_text[:100]}...")

    train_loader = create_dataloader(
        csv_path="recognition/fineTuneRadiology_48543200/data/train.csv",
        tokenizer=tokenizer,
        batch_size=8,
        shuffle=True,
    )

    print("\n✅ DataLoader ready for custom training loop!")
