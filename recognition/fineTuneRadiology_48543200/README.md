# Radiology Report → Lay Summary with FLAN-T5 + LoRA (Task 13)

**Student ID:** 38543200  
**Branch:** topic-recognition  
**Folder:** `recognition/fineTuneRadiology_48543200`

## Problem
Translate expert radiology reports into layperson summaries using the BioLaySumm 2025 open-source track.

## Model & Params
- Base: `google/flan-t5-base`
- Fine-tuning: **LoRA (PEFT)**, r=16, alpha=32, dropout=0.1
- Parameter count (trainable): ~ a few million (adapters only)
- Full params (frozen base): ~250M

## Environment
- GPU: (e.g., Colab T4 / A100)
- VRAM: (e.g., 15 GB)
- Training time: (fill after run)
- Epochs / LR / BS: 3 / 3e-4 / 4

## Usage
```bash
python train.py --use_lora --epochs 3 --batch_size 4 --lr 3e-4
python predict.py --model_dir ./outputs_flan_t5_lora --num_examples 5
