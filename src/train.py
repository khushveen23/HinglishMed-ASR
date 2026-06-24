"""
Pillar 2 — Advanced Training: Tokenizer Expansion + LoRA/QLoRA + Prompt Conditioning
======================================================================================
Supports three training modes (controlled by --mode flag):
  1. full_finetune   — Full fine-tune of whisper-small (baseline, matches original notebook)
  2. lora            — LoRA on whisper-medium (parameter-efficient, recommended)
  3. qlora           — QLoRA (4-bit quantised) on whisper-medium (memory-efficient)

Key upgrades over the original notebook:
  • Tokenizer vocabulary expansion with medical-Hinglish tokens
  • LoRA / QLoRA via PEFT library
  • Whisper prompt conditioning (decoder prefix with medical context)
  • W&B experiment tracking
  • Proper train/val/test split handling

Usage:
    python src/train.py \
        --manifest_path data/manifest_augmented.json \
        --output_dir    models/lora_medium \
        --mode          lora \
        --model_name    openai/whisper-medium \
        --wandb_project MedASR_Masters
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import evaluate

# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2a — Tokenizer expansion
# ─────────────────────────────────────────────────────────────────────────────

# Medical-Hinglish tokens that Whisper's BPE tokenizer fragments sub-optimally.
# Adding them as atomic tokens reduces WER on clinical terms.
MEDICAL_HINGLISH_TOKENS = [
    # Dosage patterns
    "500mg", "250mg", "1000mg", "400mg", "5mg", "10mg", "20mg", "40mg",
    "100mg", "200mg", "50mg", "2mg", "1mg", "0.5mg",
    # Abbreviations
    "BP", "HR", "SpO2", "HbA1c", "CBC", "LFT", "KFT", "TSH", "ECG", "MRI",
    "CT", "USG", "OPD", "ICU", "IV", "IM", "SC", "PO", "BD", "TDS", "OD",
    # Hinglish clinical terms
    "bukhar", "dard", "swelling", "breathlessness", "palpitations",
    "constipation", "acidity", "gastritis", "hypertension", "diabetes",
    "hypothyroidism", "dengue", "malaria", "UTI", "pneumonia",
    # Drug names (as single tokens)
    "Paracetamol", "Metformin", "Amlodipine", "Atorvastatin", "Azithromycin",
    "Amoxicillin", "Omeprazole", "Pantoprazole", "Ibuprofen", "Cetirizine",
    "Aspirin", "Clopidogrel", "Losartan", "Metoprolol", "Levothyroxine",
    "Dexamethasone", "Ondansetron", "Diazepam", "Phenytoin", "Sumatriptan",
    # Numeric clinical values
    "140/90", "120/80", "80/50", "SpO2-88", "HbA1c-8.5",
]


def expand_tokenizer(tokenizer: WhisperTokenizer, new_tokens: list[str]) -> int:
    """
    Add medical-Hinglish tokens to the tokenizer vocabulary.
    Returns the number of tokens actually added (skips existing ones).
    """
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if not tokens_to_add:
        print("All medical tokens already in vocabulary.")
        return 0
    n_added = tokenizer.add_tokens(tokens_to_add)
    print(f"Added {n_added} new tokens to tokenizer vocabulary.")
    return n_added


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2b — LoRA / QLoRA setup
# ─────────────────────────────────────────────────────────────────────────────

def apply_lora(model: WhisperForConditionalGeneration, r: int = 32, alpha: int = 64):
    """
    Apply LoRA to Whisper's attention projection layers.
    Targets: q_proj, v_proj in both encoder and decoder.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("Run: pip install peft")

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def apply_qlora(model_name: str, r: int = 16, alpha: int = 32):
    """
    Load model in 4-bit quantisation and apply LoRA (QLoRA).
    Requires bitsandbytes: pip install bitsandbytes
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except ImportError:
        raise ImportError("Run: pip install peft bitsandbytes")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 2c — Prompt conditioning
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_PROMPT = "The following is a medical transcription in Hinglish (Hindi-English code-switching)."


def get_decoder_prefix_ids(tokenizer: WhisperTokenizer, prompt: str) -> list[int]:
    """
    Encode a prompt string as decoder prefix token IDs.
    Whisper uses these as the initial decoder input to condition generation.
    """
    return tokenizer.encode(prompt, add_special_tokens=False)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_from_manifest(manifest_path: str) -> DatasetDict:
    """Load the manifest JSON into a HuggingFace DatasetDict."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    splits: dict[str, list] = {"train": [], "validation": [], "test": []}
    for item in manifest:
        split = item.get("split", "train")
        if split in splits:
            splits[split].append({"audio": item["audio_path"], "text": item["text"]})

    dataset_dict = {}
    for split_name, items in splits.items():
        if items:
            ds = Dataset.from_list(items)
            ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
            dataset_dict[split_name] = ds

    return DatasetDict(dataset_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Data collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    prompt_ids: list[int] = field(default_factory=list)

    def __call__(self, features: list[dict]) -> dict:
        # Audio → log-mel features
        input_features = [
            {"input_features": self.processor.feature_extractor(
                f["audio"]["array"],
                sampling_rate=f["audio"]["sampling_rate"],
            ).input_features[0]}
            for f in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Text → token IDs
        label_features = [
            {"input_ids": self.processor.tokenizer(f["text"]).input_ids}
            for f in features
        ]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (Whisper adds it internally)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(dataset: DatasetDict, processor: WhisperProcessor) -> DatasetDict:
    """Convert raw audio + text into model-ready features."""

    def prepare_sample(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    return dataset.map(
        prepare_sample,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
        desc="Preprocessing",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def build_compute_metrics(processor: WhisperProcessor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer * 100, 2)}

    return compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedASR training pipeline")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--mode",          choices=["full_finetune", "lora", "qlora"],
                        default="lora")
    parser.add_argument("--model_name",    default="openai/whisper-medium")
    parser.add_argument("--num_epochs",    type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps",  type=int,   default=500)
    parser.add_argument("--max_steps",     type=int,   default=-1,
                        help="Override num_epochs if set")
    parser.add_argument("--lora_r",        type=int,   default=32)
    parser.add_argument("--lora_alpha",    type=int,   default=64)
    parser.add_argument("--use_prompt",    action="store_true",
                        help="Enable Whisper decoder prompt conditioning")
    parser.add_argument("--expand_vocab",  action="store_true",
                        help="Add medical-Hinglish tokens to tokenizer")
    parser.add_argument("--wandb_project", default=None,
                        help="W&B project name (set to enable tracking)")
    parser.add_argument("--fp16",          action="store_true")
    args = parser.parse_args()

    # ── W&B setup ────────────────────────────────────────────────────────────
    report_to = "none"
    if args.wandb_project:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.mode}_{Path(args.model_name).name}",
            )
            report_to = "wandb"
            print(f"W&B tracking enabled: project={args.wandb_project}")
        except ImportError:
            print("wandb not installed — tracking disabled. Run: pip install wandb")

    # ── Load tokenizer & feature extractor ───────────────────────────────────
    print(f"\nLoading processor for {args.model_name}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.model_name, language="Hindi", task="transcribe"
    )

    # ── Pillar 2a: Tokenizer expansion ───────────────────────────────────────
    n_new_tokens = 0
    if args.expand_vocab:
        n_new_tokens = expand_tokenizer(tokenizer, MEDICAL_HINGLISH_TOKENS)

    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    raw_dataset = load_dataset_from_manifest(args.manifest_path)
    print(raw_dataset)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading model ({args.mode})...")
    if args.mode == "qlora":
        model = apply_qlora(args.model_name, r=args.lora_r, alpha=args.lora_alpha)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
        if n_new_tokens > 0:
            # Resize embedding layer to accommodate new tokens
            model.resize_token_embeddings(len(tokenizer))
        if args.mode == "lora":
            model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)

    # ── Pillar 2c: Prompt conditioning ───────────────────────────────────────
    if args.use_prompt:
        prompt_ids = get_decoder_prefix_ids(tokenizer, MEDICAL_PROMPT)
        model.config.forced_decoder_ids = None  # disable language forcing
        model.generation_config.prompt_ids = prompt_ids
        print(f"Prompt conditioning enabled: '{MEDICAL_PROMPT}'")
    else:
        # Standard Hinglish transcription conditioning
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="hindi", task="transcribe"
        )

    model.config.suppress_tokens = []

    # ── Data collator ────────────────────────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── Training arguments ───────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs if args.max_steps == -1 else None,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=report_to,
        push_to_hub=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset.get("validation"),
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nStarting training (mode={args.mode})...")
    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    trainer.save_model(str(output_dir / "best_model"))
    processor.save_pretrained(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))
    print(f"\nModel saved → {output_dir / 'best_model'}")

    if args.wandb_project:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
