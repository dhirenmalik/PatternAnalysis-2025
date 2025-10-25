# predict.py (updated for Task 13 - Radiology → Lay Summary)
import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import PeftModel
import evaluate

from dataset import load_biolaysumm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs_flan_t5_lora")
    p.add_argument("--num_examples", type=int, default=5)
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--add_prefix", default="summarize radiology: ")
    p.add_argument("--rouge_samples", type=int, default=500, help="Number of validation samples for ROUGE eval")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Allow local folder
    model_dir = args.model_dir.lstrip("./")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model = base_model.to(device)
    model.eval()

    # Load dataset
    raw = load_biolaysumm()
    test = raw["test"]
    cols = list(test.features.keys())
    in_col = next((c for c in cols if "report" in c.lower() or "source" in c.lower()), cols[0])
    tgt_col = next((c for c in cols if "summary" in c.lower() or "target" in c.lower()), cols[-1])

    print(f"Using columns -> input: {in_col} | target: {tgt_col}")

    # ----------------------------
    # Example qualitative outputs
    # ----------------------------
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

    # ----------------------------
    # Quantitative ROUGE evaluation
    # ----------------------------
    print("\n📊 Computing ROUGE metrics on subset...")
    rouge = evaluate.load("rouge")

    subset = raw["validation"].select(range(min(args.rouge_samples, len(raw["validation"]))))
    preds, refs = [], []

    for i in range(len(subset)):
        example = subset[i]
        text_in = args.add_prefix + example[in_col]
        inputs = tokenizer(text_in, return_tensors="pt", truncation=True,
                           padding=True, max_length=args.max_input_len).to(
            device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=args.max_target_len)
        preds.append(tokenizer.decode(output[0], skip_special_tokens=True))
        refs.append(example[tgt_col])

    results = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    results = {k: round(v, 4) for k, v in results.items()}

    print("\n✅ ROUGE Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "RESULTS.txt"), "a") as f:
        f.write("\n\nROUGE evaluation:\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print(f"\n📄 ROUGE results appended to {os.path.join(model_dir, 'RESULTS.txt')}")

if __name__ == "__main__":
    main()
