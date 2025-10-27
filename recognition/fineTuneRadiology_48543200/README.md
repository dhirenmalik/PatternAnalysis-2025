# Radiology Report → Lay Summary with FLAN-T5 + LoRA

**Student ID:** 48543200  
**Branch:** topic-recognition  
**Folder:** `recognition/fineTuneRadiology_48543200`

---

## Overview

We fine-tune the encoder–decoder language model **FLAN-T5 base** with parameter-efficient **LoRA adapters** to rewrite radiology reports into layperson summaries. This workflow targets the **BioLaySumm 2025 open-source track**, where the challenge is to translate expert clinical language into accessible text while preserving key medical facts. The automated summariser helps radiologists and hospitals rapidly produce patient-friendly explanations, reducing the manual editing burden and improving communication.

<p align="center">
  <em><!-- PLACEHOLDER: insert schematic/figure path when ready --></em>
</p>

---

## How It Works

1. **Data Preparation** – `data_setup.py` downloads BioLaySumm, normalises column names to `report_text` / `lay_summary`, and saves train/val/test CSVs. The CLI supports optional subsetting for quick iterations.
2. **Model Initialisation** – `train.py` loads FLAN-T5 base, attaches LoRA adapters (rank 16, α 32, dropout 0.1) and freezes the base weights. All defaults live in `utils.py` so they can be tuned centrally.
3. **Training Loop** – We run a custom PyTorch loop with mixed precision, gradient accumulation, cosine LR schedule with warm-up (configurable), batch-level loss logging, and periodic ROUGE evaluation. Checkpoints and plots are written to the output directory.
4. **Evaluation / Inference** – `predict.py` produces batched summaries for a chosen split and can optionally score a Hugging Face validation slice. `predict_single.py` handles ad-hoc inputs.

```
DATASET: BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track
BASE MODEL: google/flan-t5-base → 247,577,856 parameters (LoRA trainable: 1,769,472 ≈ 0.71%)
DEFAULT TRAIN CONFIG: epochs=3, batch_size=4, lr=3e-4, cosine scheduler (warmup 3%)
TRAIN SPLIT: 150,454 | VAL: 10,000 | TEST: ___ (fill once confirmed)
```

---

## Dependencies & Environment

All dependencies are pinned in [`requirements.txt`](../../requirements.txt). Install via:

```bash
pip install -r requirements.txt
```

Key packages (tested on Python 3.12):

- `torch>=2.1.0` (tested on 2.8.0+cu126 for A100 runs)  
- `transformers>=4.38.0` for FLAN-T5 and schedulers  
- `peft>=0.7.0` for LoRA adapters  
- `datasets>=2.14.0`, `evaluate>=0.4.0`, `rouge-score`, `nltk` for data + ROUGE  
- `matplotlib>=3.7.0` for loss / ROUGE / LR plots  

Reproducibility: training seeds defaults to 42; set `--seed` if you need a different run.

---

## Pre-processing & Splits

- `data_setup.py` maps dataset columns to `report_text` (input) and `lay_summary` (target); blank entries are removed automatically by the Hugging Face dataset loader.  
- Train/val/test splits follow the BioLaySumm release. For quick experiments the CLI supports `--subset N` (train ≈ N, val ≈ N/10).  
- Additional light preprocessing (e.g., truncation to `max_input_len` and `max_target_len`) happens inside the dataset class just before tokenisation—**no custom cleaning or anonymisation is required** because BioLaySumm is already de-identified.

*Justification of splits:* Official train/validation/test partitions preserve the competition protocol. When `--subset` is used, we keep a 10:1 ratio for dev testing to maintain a representative validation slice while speeding up iterations.

---

## Usage

> **Defaults:** Unless overridden, `train.py` enables LoRA, mixed precision (on CUDA), and a cosine learning-rate schedule with 3% warm-up. Use `--scheduler none` or `--no_lora` to opt out.

### Training (default LoRA fine-tune)

```bash
python recognition/fineTuneRadiology_48543200/train.py \
  --output_dir outputs_flan_t5_lora \
  --epochs 3 \
  --batch_size 4 \
  --lr 3e-4
```

### Dry Run (small subset, cosine schedule, no AMP – mac-friendly)

```bash
python recognition/fineTuneRadiology_48543200/train.py \
  --output_dir outputs_dryrun_scheduler \
  --epochs 1 \
  --batch_size 2 \
  --subset 2048 \
  --max_input_len 512 \
  --max_target_len 128 \
  --logging_steps 10 \
  --lr 1e-4 \
  --scheduler cosine \
  --warmup_ratio 0.1 \
  --no_mixed_precision
```

### Full Fine-tune (no LoRA – heavy!)

```bash
python recognition/fineTuneRadiology_48543200/train.py \
  --output_dir outputs_full_finetune \
  --epochs 3 \
  --batch_size 1 \
  --lr 1e-5 \
  --no_lora
```

### Batched Evaluation

```bash
python recognition/fineTuneRadiology_48543200/predict.py \
  --checkpoint outputs_flan_t5_lora/best_model \
  --data_dir recognition/fineTuneRadiology_48543200/data \
  --split val \
  --batch_size 8 \
  --num_examples 5 
```

### Single Example

```bash
python recognition/fineTuneRadiology_48543200/predict_single.py \
  --checkpoint outputs_flan_t5_lora/best_model \
  --text "The chest shows significant air trapping..."
```

---

## Experiment Tracking & Outputs

Each training run writes the following to `--output_dir`:

- `best_model/` and `checkpoint-step-*` directory with adapters/tokenizer + `training_state.pt`  
- `training_log.json` (loss/ROUGE/LR history, runtime, hyperparameters)  
- `training_report.txt` (human-readable summary)  
- Plots:  `loss_curve.png`, `batch_loss_curve.png`, `rouge_curves.png`, `learning_rate_curve.png`

Evaluation runs produce:

- `evaluation/predictions.csv` (input/report/prediction triplets)  
- `evaluation/rouge_scores.json`  
- `evaluation/examples.txt`  (sample outputs)  
- `evaluation_report.txt`

---

## Example Input / Output

> **Input (excerpt)**  
> "summarize for a layperson: The chest shows significant air trapping. Bilateral apical chronic changes are present. Dorsal kyphosis is noted. No evidence of pneumothorax."

> **Model Summary**  
> "The chest shows significant air trapping. There are long-term changes in the apical area. There is no sign of kyphosis."

> **Reference Summary**  
> "The chest shows a large amount of trapped air. There are long-term changes at the top of both lungs. The upper back is curved outward. There is no sign of air in the space around the lungs."

![Training loss curves](<!-- PLACEHOLDER: add plot path -->)

![ROUGE curves](<!-- PLACEHOLDER: add plot path -->)

---

## Results Snapshot

| Split  | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|--------|---------|---------|---------|-------------|
| *val (subset 4k)* | 0.5796  | 0.3481  | 0.5144  | 0.5428 |
| *validation (HF 256 samples)* | 0.5790  | 0.3368  | 0.5046  | 0.5392 |
| *val (subset 2k, cosine dry-run)* | 0.5217 | 0.2776 | 0.4548 | 0.4825 |

*(Update table once full runs complete.)*

---

## Future Work / Notes

- Add data augmentation or domain-specific lexical simplification modules.  
- Investigate full fine-tuning vs LoRA trade-offs on GPU memory/time.  
- Integrate human evaluation metrics or readability scores.

---

## References

- [BioLaySumm 2025 Dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track)  
- [RAFI et al., “Lay Summaries for Radiology Reports”]( ) <!-- TODO: add citation -->  
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## TODO / Fill-in Blanks

- Training duration on full dataset: ____  
- Final test-set performance: ____  
- Figure path(s): ____  
- Additional preprocessing steps (if any): ____
