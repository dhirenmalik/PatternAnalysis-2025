"""Download and clean the BioLaySumm2025 dataset for local training.

The script produces CSV splits (train/validation/test) containing the columns
`report_text` and `lay_summary`, while printing styled progress output.
"""

import os
from pathlib import Path
from textwrap import shorten

import pandas as pd
from datasets import load_dataset
from utils import DataParams

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CONFIG = DataParams()


SECTION: str = "=" * 70
SUBSECTION: str = "-" * 70


def print_header(title: str) -> None:
    """Print a centred title surrounded by divider lines.

    Args:
        title: Text to display in the header block.

    Returns:
        None. Output is printed directly to stdout.
    """
    print(f"\n{SECTION}\n{title:^70}\n{SECTION}")


def print_section(title: str) -> None:
    """Print a left-aligned section heading with surrounding rules.

    Args:
        title: The section title to print.

    Returns:
        None. Output is printed directly to stdout.
    """
    print(f"\n{SUBSECTION}\n{title}\n{SUBSECTION}")


def preview(series: pd.Series, width: int = 80) -> str:
    """Return a compact preview of the first non-null entry in a series.

    Args:
        series: Source pandas Series to summarise.
        width: Desired display width for the preview string.

    Returns:
        The first non-null entry truncated to the specified width, or ``"n/a"``
        if no values are available.
    """
    if series.empty:
        return "n/a"
    first_value = series.dropna().astype(str).iloc[0].replace("\n", " ").strip()
    return shorten(first_value, width=width, placeholder="...")


def main() -> None:
    """Download, clean, and export BioLaySumm2025 splits.

    Returns:
        None. Entry point handles file system side effects.
    """
    output_dir = Path(CONFIG.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("BioLaySumm2025 Dataset Preparation")
    print(f"📂 Output directory: {output_dir.resolve()}")

    print_section("Step 1 · Download dataset")
    print(f"🔽 Fetching `{CONFIG.hf_dataset_name}` from Hugging Face...")
    dataset = load_dataset(CONFIG.hf_dataset_name)

    print("\n📊 Split overview:")
    for split in ["train", "validation", "test"]:
        count = len(dataset[split])
        print(f"  • {split.capitalize():<11}{count:>8,} rows")

    split_mappings = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}

    print_section("Step 2 · Clean & export")
    for split_name, file_name in split_mappings.items():
        df = dataset[split_name].to_pandas()
        df_clean = df.rename(
            columns={"radiology_report": "report_text", "layman_report": "lay_summary"}
        )[["report_text", "lay_summary"]]

        output_path = output_dir / file_name
        df_clean.to_csv(output_path, index=False)

        print(f"\n✅ Saved `{file_name}` with {len(df_clean):,} rows")
        print(f"   ├─ Columns: {', '.join(df_clean.columns)}")
        print(f"   ├─ Sample report_text: {preview(df_clean['report_text'])}")
        print(f"   └─ Sample lay_summary: {preview(df_clean['lay_summary'])}")

    print_section("Step 3 · Summary")
    print("🎉 All splits exported successfully. Files available at:")
    for file_name in split_mappings.values():
        print(f"  - {output_dir / file_name}")


if __name__ == "__main__":
    main()
