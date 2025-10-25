# dataset.py
from typing import Dict, Tuple, Optional
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

# FIXED: Put the actual column names FIRST, metadata columns LAST
CANDIDATE_INPUT_KEYS = ("radiology_report", "expert_report", "report", "source")
CANDIDATE_TARGET_KEYS = ("layman_report", "lay_summary", "summary", "target")


def _guess_column(names, candidates):
    names_lower = [n.lower() for n in names]
    for c in candidates:
        if c.lower() in names_lower:
            return names[names_lower.index(c.lower())]
    # Fallback to the first/second column heuristics
    return names[0]


def load_biolaysumm(
        split_mapping: Optional[Dict[str, str]] = None,
) -> DatasetDict:
    """
    Loads the BioLaySumm 2025 open-source track dataset from Hugging Face.
    If the dataset has a single 'train' split, we will create validation/test from it.
    """
    ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    if isinstance(ds, DatasetDict) and set(ds.keys()) >= {"train", "validation",
                                                          "test"}:
        return ds

    # Otherwise, make validation/test splits from 'train'
    base = ds["train"] if "train" in ds else list(ds.values())[0]
    tmp = base.train_test_split(test_size=0.1, seed=42)
    val_test = tmp["test"].train_test_split(test_size=0.5, seed=42)
    return DatasetDict(train=tmp["train"], validation=val_test["train"],
                       test=val_test["test"])


def build_tokenized(
        raw: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        max_input_len: int = 1024,
        max_target_len: int = 256,
        input_col: Optional[str] = None,
        target_col: Optional[str] = None,
        add_prefix: str = "summarize radiology: ",
) -> Tuple[DatasetDict, str, str]:
    """
    Tokenizes the dataset for seq2seq. Returns (tokenized_dataset, input_col_used, target_col_used).
    """
    # Guess columns when not provided:
    sample = raw["train"].features
    cols = list(sample.keys())
    input_col = input_col or _guess_column(cols, CANDIDATE_INPUT_KEYS)
    target_col = target_col or _guess_column(cols, CANDIDATE_TARGET_KEYS)

    # Debug: Print what columns we're using
    print(f"🔍 Using columns: input='{input_col}' | target='{target_col}'")

    def preprocess(batch):
        sources = [add_prefix + s for s in batch[input_col]]
        model_inputs = tokenizer(
            sources, max_length=max_input_len, truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            batch[target_col], max_length=max_target_len, truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=cols)
    return tokenized, input_col, target_col