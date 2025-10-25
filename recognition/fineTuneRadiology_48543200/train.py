# train.py
# COMP3710 Task 13 (Radiology → Lay Summary)
# Student ID: 48543200

import argparse
import os
import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

from modules import load_model_and_tokenizer, ModelConfig
from dataset import load_biolaysumm, build_tokenized


def parse_args():
    p = argparse.ArgumentParser()

    # existing args
    p.add_argument("--model_name", default="google/flan-t5-base")
    p.add_argument("--output_dir", default="./outputs_flan_t5_lora")
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=256)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)

    # NEW args for fast Colab debugging
    p.add_argument("--subset", type=int, default=None,
                   help="If set, use only this many samples from train and ~10% of that for val.")
    p.add_argument("--fp16", action="store_true",
                   help="Force fp16 mixed precision.")
    p.add_argument("--checkpointing", action="store_true",
                   help="Enable gradient checkpointing (only works without LoRA).")
    p.add_argument("--eval_epoch", action="store_true",
                   help="Evaluate only at end of each epoch.")
    p.add_argument("--save_epoch", action="store_true",
                   help="Save only at end of each epoch.")

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------
    # 1) Model + tokenizer
    # -----------------------------
    cfg = ModelConfig(model_name=args.model_name, use_lora=args.use_lora)
    model, tokenizer, _ = load_model_and_tokenizer(cfg)

    # FIXED: Only enable gradient checkpointing if NOT using LoRA
    if args.checkpointing:
        if args.use_lora:
            print(
                "⚠️  WARNING: Gradient checkpointing is incompatible with LoRA.")
            print("    Disabling checkpointing to avoid training errors.")
            print(
                "    To save VRAM: reduce batch_size or max_input_len instead.")
        else:
            print("🔁 Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()

    # -----------------------------
    # 2) Data
    # -----------------------------
    raw = load_biolaysumm()

    tokenized, input_col, target_col = build_tokenized(
        raw,
        tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

    # OPTIONAL SUBSET for fast iteration
    if args.subset is not None:
        n_train = min(args.subset, len(tokenized["train"]))
        n_val = max(1, args.subset // 10)
        n_val = min(n_val, len(tokenized["validation"]))

        print(f"⚙️ Using subset: {n_train} train / {n_val} val (from full "
              f"{len(tokenized['train'])} / {len(tokenized['validation'])})")

        tokenized_train = tokenized["train"].select(range(n_train))
        tokenized_val = tokenized["validation"].select(range(n_val))
    else:
        tokenized_train = tokenized["train"]
        tokenized_val = tokenized["validation"]

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -----------------------------
    # 3) Metrics (ROUGE-1/2/L/Lsum)
    # -----------------------------
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels,
                                                skip_special_tokens=True)

        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in
                    preds]
        results["gen_len"] = float(np.mean(gen_lens))
        return {k: round(v, 4) for k, v in results.items()}

    # -----------------------------
    # 4) Trainer / TrainingArguments
    # -----------------------------

    eval_strategy = "epoch" if args.eval_epoch else "steps"
    save_strategy = "epoch" if args.save_epoch else "steps"

    use_fp16 = args.fp16 or torch.cuda.is_available()

    # Debug: Print dataset sizes
    print(f"📊 Training samples: {len(tokenized_train)}")
    print(f"📊 Validation samples: {len(tokenized_val)}")
    print(f"📊 Steps per epoch: {len(tokenized_train) // args.batch_size}")
    print(
        f"📊 Total training steps: {(len(tokenized_train) // args.batch_size) * args.epochs}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,

        # logging / eval / save
        evaluation_strategy=eval_strategy,
        save_strategy=save_strategy,
        eval_steps=args.eval_steps if eval_strategy == "steps" else None,
        save_steps=args.save_steps if save_strategy == "steps" else None,
        logging_steps=args.logging_steps,

        # generation config for eval
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,

        # stability / perf
        fp16=use_fp16,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,

        # misc / reporting
        report_to=["none"],
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # -----------------------------
    # 5) Train + evaluate
    # -----------------------------
    print("✅ Training setup complete — starting fine-tuning...")
    trainer.train()
    print("🏁 Training done, running final eval on validation split...")
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)

    # -----------------------------
    # 6) Save model + tokenizer + results text
    # -----------------------------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "RESULTS.txt"), "w") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"📦 Saved model + tokenizer + RESULTS.txt to {args.output_dir}")


if __name__ == "__main__":
    main()