# train.py
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
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Model + tokenizer
    cfg = ModelConfig(model_name=args.model_name, use_lora=args.use_lora)
    model, tokenizer, _ = load_model_and_tokenizer(cfg)

    # 2) Data
    raw = load_biolaysumm()
    tokenized, input_col, target_col = build_tokenized(
        raw,
        tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 3) Metrics (ROUGE-1/2/L/Lsum)
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )
        # Also report average generation length
        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        results["gen_len"] = np.mean(gen_lens)
        return {k: round(v, 4) for k, v in results.items()}

    # 4) Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        fp16=torch.cuda.is_available(),
        report_to=["none"],
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5) Train + evaluate
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)

    # 6) Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Simple text report (you can paste this into your README later)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "RESULTS.txt"), "w") as f:
        for k, v in eval_metrics.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()
