"""
Pillar 3, 4 & 5 — Clinical Evaluation, Post-Processing & MLOps
================================================================
Implements:
  • Standard WER (baseline)
  • cWER — Clinical Word Error Rate (Pillar 3)
  • Semantic evaluation via BLEU / ROUGE (Pillar 3)
  • Inverse Text Normalisation (ITN) post-processing (Pillar 4)
  • LM re-ranking of beam-search hypotheses (Pillar 4)
  • W&B artifact logging of evaluation results (Pillar 5)
  • Comparison table: baseline vs fine-tuned vs LoRA (Pillar 5)

Usage:
    python src/evaluate.py \
        --manifest_path  data/manifest.json \
        --model_dir      models/lora_medium/best_model \
        --baseline_model openai/whisper-small \
        --output_dir     results \
        --split          test \
        --wandb_project  MedASR_Masters
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Clinical entity vocabulary for cWER
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_ENTITY_PATTERNS = [
    # Dosages: 500mg, 10mg, 1000mg, etc.
    r"\b\d+\s*mg\b",
    # BP readings: 140/90, 120/80
    r"\b\d{2,3}/\d{2,3}\b",
    # Percentages: 8.5%, 88%
    r"\b\d+\.?\d*\s*%",
    # Drug names (English)
    r"\b(?:Paracetamol|Metformin|Amlodipine|Atorvastatin|Azithromycin|"
    r"Amoxicillin|Omeprazole|Pantoprazole|Ibuprofen|Cetirizine|Aspirin|"
    r"Clopidogrel|Losartan|Metoprolol|Levothyroxine|Dexamethasone|"
    r"Ondansetron|Diazepam|Phenytoin|Sumatriptan|Insulin)\b",
    # Lab abbreviations
    r"\b(?:BP|HR|SpO2|HbA1c|CBC|LFT|KFT|TSH|ECG|MRI|CT|USG|UTI|ICU|IV)\b",
    # Critical symptoms (English component)
    r"\b(?:chest pain|breathlessness|hypoglycemia|tachycardia|seizure|stroke)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CLINICAL_ENTITY_PATTERNS]


def extract_clinical_entities(text: str) -> list[str]:
    """Extract all clinical entity tokens from a text string."""
    entities = []
    for pattern in COMPILED_PATTERNS:
        entities.extend(pattern.findall(text))
    return [e.lower().strip() for e in entities]


def compute_cwer(references: list[str], hypotheses: list[str]) -> dict:
    """
    Clinical Word Error Rate: WER computed only on clinical entity tokens.
    Returns overall cWER and per-category breakdown.
    """
    from jiwer import wer as jiwer_wer

    ref_entities = [" ".join(extract_clinical_entities(r)) for r in references]
    hyp_entities = [" ".join(extract_clinical_entities(h)) for h in hypotheses]

    # Filter out pairs where reference has no clinical entities
    valid_pairs = [(r, h) for r, h in zip(ref_entities, hyp_entities) if r.strip()]
    if not valid_pairs:
        return {"cwer": None, "n_clinical_pairs": 0}

    valid_refs, valid_hyps = zip(*valid_pairs)
    cwer = jiwer_wer(list(valid_refs), list(valid_hyps))

    return {
        "cwer": round(cwer * 100, 2),
        "n_clinical_pairs": len(valid_pairs),
        "total_pairs": len(references),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 3 — Semantic evaluation (BLEU / ROUGE)
# ─────────────────────────────────────────────────────────────────────────────

def compute_semantic_metrics(references: list[str], hypotheses: list[str]) -> dict:
    """Compute BLEU-4 and ROUGE-L to measure semantic preservation."""
    try:
        import evaluate as hf_evaluate
    except ImportError:
        raise ImportError("Run: pip install evaluate rouge_score sacrebleu")

    bleu  = hf_evaluate.load("sacrebleu")
    rouge = hf_evaluate.load("rouge")

    bleu_result  = bleu.compute(
        predictions=hypotheses,
        references=[[r] for r in references],
    )
    rouge_result = rouge.compute(
        predictions=hypotheses,
        references=references,
    )

    return {
        "bleu4":   round(bleu_result["score"], 2),
        "rouge_l": round(rouge_result["rougeL"] * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — Inverse Text Normalisation (ITN)
# ─────────────────────────────────────────────────────────────────────────────

# Hindi number words → digits
HINDI_NUMBERS = {
    "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5",
    "पांच": "5", "छह": "6", "सात": "7", "आठ": "8", "नौ": "9",
    "दस": "10", "बीस": "20", "तीस": "30", "चालीस": "40", "पचास": "50",
    "साठ": "60", "सत्तर": "70", "अस्सी": "80", "नब्बे": "90",
    "सौ": "100", "हज़ार": "1000", "हजार": "1000",
    "एक सौ": "100", "दो सौ": "200", "पाँच सौ": "500",
    "एक सो चालीस": "140", "नब्बे": "90",
}

# Spoken BP patterns → standard notation
BP_SPOKEN_PATTERNS = [
    (r"(\d+)\s+ओवर\s+(\d+)", r"\1/\2"),
    (r"(\d+)\s+over\s+(\d+)", r"\1/\2"),
    (r"(\d+)\s+बटा\s+(\d+)", r"\1/\2"),
]

# Spoken dosage patterns
DOSAGE_PATTERNS = [
    (r"(\d+)\s+एमजी", r"\1mg"),
    (r"(\d+)\s+mg", r"\1mg"),
    (r"(\d+)\s+milligram", r"\1mg"),
]

# Percentage patterns
PERCENT_PATTERNS = [
    (r"(\d+\.?\d*)\s+परसेंट", r"\1%"),
    (r"(\d+\.?\d*)\s+percent", r"\1%"),
]


def apply_itn(text: str) -> str:
    """
    Apply Inverse Text Normalisation to convert spoken forms to standard
    clinical notation.
    """
    result = text

    # Replace Hindi number words
    for word, digit in sorted(HINDI_NUMBERS.items(), key=lambda x: -len(x[0])):
        result = re.sub(r"\b" + re.escape(word) + r"\b", digit, result, flags=re.IGNORECASE)

    # Normalise BP readings
    for pattern, replacement in BP_SPOKEN_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Normalise dosages
    for pattern, replacement in DOSAGE_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Normalise percentages
    for pattern, replacement in PERCENT_PATTERNS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    return result.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pillar 4 — LM re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def lm_rerank_hypotheses(
    hypotheses: list[str],
    lm_model_name: str = "ai4bharat/indic-bert",
) -> str:
    """
    Score a list of beam-search hypotheses with a language model and return
    the highest-scoring one.

    Uses pseudo-perplexity (sum of log-probs) from a masked LM as a proxy
    for clinical plausibility.
    """
    if not hypotheses:
        return ""
    if len(hypotheses) == 1:
        return hypotheses[0]

    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
    except ImportError:
        return hypotheses[0]  # fallback: return first hypothesis

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cache model across calls
    if not hasattr(lm_rerank_hypotheses, "_model"):
        print(f"Loading LM re-ranker: {lm_model_name}")
        lm_rerank_hypotheses._tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
        lm_rerank_hypotheses._model = AutoModelForMaskedLM.from_pretrained(
            lm_model_name
        ).to(device).eval()

    tokenizer = lm_rerank_hypotheses._tokenizer
    model     = lm_rerank_hypotheses._model

    scores = []
    for hyp in hypotheses:
        inputs = tokenizer(hyp, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # Lower loss = higher probability = better hypothesis
        scores.append(-outputs.loss.item())

    best_idx = int(np.argmax(scores))
    return hypotheses[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_dataset(
    manifest: list[dict],
    model_dir: str,
    use_itn: bool = True,
    use_lm_rerank: bool = False,
    num_beams: int = 5,
    batch_size: int = 8,
) -> list[dict]:
    """
    Run inference on the dataset and return a list of result dicts.
    Applies ITN and optionally LM re-ranking.
    """
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} on {device}...")

    # Handle PEFT / LoRA models
    try:
        from peft import PeftModel, PeftConfig
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model  = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path
        )
        model = PeftModel.from_pretrained(base_model, model_dir).to(device)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
        print("Loaded LoRA/PEFT model.")
    except Exception:
        model     = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
        processor = WhisperProcessor.from_pretrained(model_dir)
        print("Loaded standard model.")

    model.eval()

    results = []
    for item in tqdm(manifest, desc="Transcribing"):
        try:
            wav, sr = librosa.load(item["audio_path"], sr=16_000, mono=True)
            inputs  = processor(wav, sampling_rate=16_000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                generated = model.generate(
                    input_features,
                    num_beams=num_beams,
                    num_return_sequences=num_beams if use_lm_rerank else 1,
                    language="hi",
                    task="transcribe",
                )

            if use_lm_rerank and num_beams > 1:
                hyps = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
                prediction = lm_rerank_hypotheses(hyps)
            else:
                prediction = processor.tokenizer.decode(
                    generated[0], skip_special_tokens=True
                )

            if use_itn:
                prediction = apply_itn(prediction)

            results.append({
                "audio_path": item["audio_path"],
                "reference":  item["text"],
                "hypothesis": prediction,
                "speaker_id": item.get("speaker_id", "unknown"),
                "split":      item.get("split", "test"),
            })
        except Exception as e:
            print(f"\nError on {item['audio_path']}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    manifest: list[dict],
    model_dir: str,
    model_label: str,
    use_itn: bool = True,
    use_lm_rerank: bool = False,
) -> dict:
    """Run full evaluation and return metrics dict."""
    from jiwer import wer as jiwer_wer

    results = transcribe_dataset(manifest, model_dir, use_itn=use_itn,
                                 use_lm_rerank=use_lm_rerank)

    refs = [r["reference"]  for r in results]
    hyps = [r["hypothesis"] for r in results]

    # Standard WER
    std_wer = jiwer_wer(refs, hyps) * 100

    # Clinical WER
    cwer_result = compute_cwer(refs, hyps)

    # Semantic metrics
    sem_metrics = compute_semantic_metrics(refs, hyps)

    metrics = {
        "model":    model_label,
        "wer":      round(std_wer, 2),
        "cwer":     cwer_result.get("cwer"),
        "bleu4":    sem_metrics["bleu4"],
        "rouge_l":  sem_metrics["rouge_l"],
        "n_samples": len(results),
    }

    return metrics, results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedASR evaluation pipeline")
    parser.add_argument("--manifest_path",  required=True)
    parser.add_argument("--model_dir",      required=True,
                        help="Fine-tuned model directory")
    parser.add_argument("--baseline_model", default="openai/whisper-small",
                        help="Baseline model for comparison")
    parser.add_argument("--output_dir",     default="results")
    parser.add_argument("--split",          default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--use_itn",        action="store_true",
                        help="Apply Inverse Text Normalisation")
    parser.add_argument("--use_lm_rerank",  action="store_true",
                        help="Apply LM re-ranking on beam hypotheses")
    parser.add_argument("--num_beams",      type=int, default=5)
    parser.add_argument("--wandb_project",  default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load manifest ────────────────────────────────────────────────────────
    with open(args.manifest_path) as f:
        full_manifest = json.load(f)
    manifest = [m for m in full_manifest if m.get("split") == args.split]
    print(f"Evaluating on {len(manifest)} samples from '{args.split}' split.")

    # ── W&B init ─────────────────────────────────────────────────────────────
    if args.wandb_project:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=f"eval_{args.split}",
                       config=vars(args))
        except ImportError:
            print("wandb not installed.")

    all_metrics = []

    # ── Evaluate fine-tuned model ─────────────────────────────────────────────
    print("\n── Evaluating fine-tuned model ──────────────────────────────")
    ft_metrics, ft_results = evaluate_model(
        manifest, args.model_dir, model_label="fine_tuned",
        use_itn=args.use_itn, use_lm_rerank=args.use_lm_rerank,
    )
    all_metrics.append(ft_metrics)

    # Save detailed results
    ft_df = pd.DataFrame(ft_results)
    ft_df.to_csv(output_dir / "results_finetuned.csv", index=False)

    # ── Evaluate baseline ─────────────────────────────────────────────────────
    print("\n── Evaluating baseline model ────────────────────────────────")
    bl_metrics, bl_results = evaluate_model(
        manifest, args.baseline_model, model_label="baseline_whisper",
        use_itn=False, use_lm_rerank=False,
    )
    all_metrics.append(bl_metrics)

    bl_df = pd.DataFrame(bl_results)
    bl_df.to_csv(output_dir / "results_baseline.csv", index=False)

    # ── Comparison table ──────────────────────────────────────────────────────
    comparison_df = pd.DataFrame(all_metrics)
    comparison_path = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n── Evaluation Results ───────────────────────────────────────")
    print(comparison_df.to_string(index=False))
    print(f"\nComparison table saved → {comparison_path}")

    # ── Error analysis: worst-performing samples ──────────────────────────────
    from jiwer import wer as jiwer_wer
    ft_df["sample_wer"] = ft_df.apply(
        lambda row: jiwer_wer(row["reference"], row["hypothesis"]) * 100
        if row["reference"].strip() else 0.0,
        axis=1,
    )
    worst = ft_df.nlargest(20, "sample_wer")[
        ["reference", "hypothesis", "sample_wer", "speaker_id"]
    ]
    worst.to_csv(output_dir / "worst_samples.csv", index=False)
    print(f"\nTop-20 worst samples saved → {output_dir / 'worst_samples.csv'}")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if args.wandb_project:
        try:
            import wandb
            for m in all_metrics:
                wandb.log({f"{m['model']}/{k}": v for k, v in m.items() if k != "model"})

            # Log comparison table as W&B artifact
            artifact = wandb.Artifact("evaluation_results", type="evaluation")
            artifact.add_file(str(comparison_path))
            artifact.add_file(str(output_dir / "results_finetuned.csv"))
            artifact.add_file(str(output_dir / "worst_samples.csv"))
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception as e:
            print(f"W&B logging error: {e}")

    # ── Print sample predictions ──────────────────────────────────────────────
    print("\n── Sample Predictions (Fine-tuned) ──────────────────────────")
    for _, row in ft_df.head(10).iterrows():
        print(f"  REF : {row['reference']}")
        print(f"  HYP : {row['hypothesis']}")
        print(f"  WER : {row['sample_wer']:.1f}%")
        print()


if __name__ == "__main__":
    main()
