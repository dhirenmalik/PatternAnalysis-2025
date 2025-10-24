# predict.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import PeftModel

from dataset import load_biolaysumm, build_tokenized


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./outputs_flan_t5_lora")
    p.add_argument("--num_examples", type=int, default=5)
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--add_prefix", default="summarize radiology: ")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    model = base_model.to(device)
    model.eval()

    # Load raw test split to show real examples:
    raw = load_biolaysumm()
    test = raw["test"]
    cols = list(test.features.keys())

    # Try to find likely text columns
    in_col = next((c for c in cols if "report" in c.lower() or "source" in c.lower()), cols[0])
    tgt_col = next((c for c in cols if "summary" in c.lower() or "target" in c.lower()), cols[-1])

    print(f"Using columns -> input: {in_col} | target: {tgt_col}")

    for ex in test.select(range(min(args.num_examples, len(test)))):
        prompt = args.add_prefix + ex[in_col]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_input_len).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_length=args.max_target_len,
                num_beams=4,
                length_penalty=1.0,
            )
        pred = tokenizer.decode(gen[0], skip_special_tokens=True)

        print("\n=== EXAMPLE ===")
        print("EXPERT REPORT:\n", ex[in_col][:800], "..." if len(ex[in_col]) > 800 else "")
        print("\nMODEL SUMMARY:\n", pred)
        if tgt_col in ex and isinstance(ex[tgt_col], str):
            print("\nREFERENCE SUMMARY:\n", ex[tgt_col])


if __name__ == "__main__":
    main()
