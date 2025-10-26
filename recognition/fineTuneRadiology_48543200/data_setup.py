"""
Download and prepare the BioLaySumm2025 dataset from Hugging Face.
Saves the splits as CSV files: train.csv, val.csv, test.csv
with columns: report_text, lay_summary
"""

import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def main():
    """Main entry point for preparing the BioLaySumm2025 dataset."""
    # Define output directory
    output_dir = Path("recognition/fineTuneRadiology_48543200/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔽 Downloading BioLaySumm2025 dataset from Hugging Face...")
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    # Display dataset size summary
    print("\n📊 Dataset loaded successfully:")
    for split in ["train", "validation", "test"]:
        print(f"  - {split.capitalize()}: {len(dataset[split])} samples")

    # Process each split and save to CSV
    split_mappings = {
        "train": "train.csv",
        "validation": "val.csv",
        "test": "test.csv"
    }

    for split_name, file_name in split_mappings.items():
        df = dataset[split_name].to_pandas()

        # Rename and select relevant columns
        df_clean = df.rename(
            columns={
                "radiology_report": "report_text",
                "layman_report": "lay_summary"
            }
        )[["report_text", "lay_summary"]]

        # Save the cleaned data
        output_path = output_dir / file_name
        df_clean.to_csv(output_path, index=False)

        print(f"\n✅ Saved {file_name} ({len(df_clean)} rows)")
        print(f"   Columns: {list(df_clean.columns)}")
        print(f"   Sample report_text: {df_clean['report_text'].iloc[0][:80]}...")
        print(f"   Sample lay_summary: {df_clean['lay_summary'].iloc[0][:80]}...")

    print("\n🎉 All splits saved successfully!")
    print("\n📂 File locations:")
    for file_name in split_mappings.values():
        print(f"  - {output_dir / file_name}")


if __name__ == "__main__":
    main()
