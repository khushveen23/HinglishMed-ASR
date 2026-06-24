# MedASR Masters — Hinglish Clinical Speech Recognition

A Master's-level Automatic Speech Recognition system for **Hinglish (Hindi-English code-switching) medical speech**, built on OpenAI Whisper with five research-grade upgrades.

---

## Project Structure

```
MedASR_Masters/
├── src/
│   ├── data_generation.py      # Pillar 1 — LLM dataset + multi-speaker TTS
│   ├── audio_augmentation.py   # Pillar 1 — Acoustic augmentation pipeline
│   ├── train.py                # Pillar 2 — Tokenizer expansion + LoRA/QLoRA
│   └── evaluate.py             # Pillars 3,4,5 — cWER, ITN, LM re-rank, W&B
├── MedASR_Masters_Colab.ipynb  # All-in-one Google Colab notebook
├── requirements.txt
└── README.md
```

---

## Five Research Pillars

### Pillar 1 — Scale & Realism of Data
- **LLM-driven generation**: 5,000 unique Hinglish clinical sentences via GPT-4o-mini, seeded with a curated medical dictionary (drugs, symptoms, diagnoses)
- **Multi-speaker synthesis**: Coqui XTTS-v2 renders each sentence across 20 synthetic speaker identities (varied gender, age, speaking rate)
- **Acoustic augmentation**: Gaussian noise, telephone filtering, room acoustics, time-stretch, pitch-shift

### Pillar 2 — Advanced Modeling
- **Tokenizer expansion**: Medical-Hinglish tokens (`500mg`, `HbA1c`, `BP`, drug names) added as atomic vocabulary entries
- **LoRA fine-tuning**: `whisper-medium` with Low-Rank Adaptation (r=32) — ~1% of parameters trained
- **QLoRA**: 4-bit quantised training for memory-constrained environments
- **Prompt conditioning**: Decoder prefix `"The following is a medical transcription in Hinglish..."` steers code-switching accuracy

### Pillar 3 — Clinical Evaluation Metrics
- **Standard WER**: Baseline metric
- **cWER (Clinical WER)**: WER computed exclusively on clinical entities (dosages, drug names, lab values, BP readings)
- **Semantic metrics**: BLEU-4 and ROUGE-L to measure clinical intent preservation

### Pillar 4 — Post-Processing & Hybrid Decoding
- **Inverse Text Normalisation (ITN)**: Converts spoken forms → standard notation (`एक सो चालीस ओवर नब्बे` → `140/90`, `500 एमजी` → `500mg`)
- **LM re-ranking**: Beam hypotheses scored by `ai4bharat/indic-bert` pseudo-perplexity; most clinically plausible hypothesis selected

### Pillar 5 — Production Engineering & MLOps
- **Modular codebase**: Decoupled `src/` scripts, each independently runnable
- **W&B tracking**: Loss curves, WER per checkpoint, audio artifacts, comparison tables
- **Experiment comparison**: Baseline vs full fine-tune vs LoRA in a single evaluation run

---

## Quick Start (Google Colab)

Open `MedASR_Masters_Colab.ipynb` in Google Colab with a **T4 GPU** runtime.

The notebook walks through all five pillars end-to-end with inline explanations.

---

## Manual Pipeline

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 5,000 sentences + synthesise audio
python src/data_generation.py \
    --output_dir data \
    --n_sentences 5000 \
    --openai_api_key YOUR_KEY \
    --use_xtts

# 3. Augment audio
python src/audio_augmentation.py \
    --manifest_path data/manifest.json \
    --output_dir    data/audio_augmented \
    --augment_train_only

# 4. Train with LoRA
python src/train.py \
    --manifest_path data/manifest_augmented.json \
    --output_dir    models/lora_medium \
    --mode          lora \
    --model_name    openai/whisper-medium \
    --expand_vocab \
    --use_prompt \
    --wandb_project MedASR_Masters

# 5. Evaluate
python src/evaluate.py \
    --manifest_path  data/manifest.json \
    --model_dir      models/lora_medium/best_model \
    --baseline_model openai/whisper-small \
    --output_dir     results \
    --use_itn \
    --wandb_project  MedASR_Masters
```

---

## Experiment Comparison

| Model | WER ↓ | cWER ↓ | BLEU-4 ↑ | ROUGE-L ↑ |
|---|---|---|---|---|
| Whisper-small (baseline) | ~45% | ~62% | ~18 | ~42 |
| Full fine-tune (whisper-small) | ~22% | ~28% | ~51 | ~68 |
| LoRA (whisper-medium) | ~14% | ~17% | ~64 | ~79 |
| LoRA + ITN + LM re-rank | ~11% | ~13% | ~68 | ~83 |

*Numbers are indicative targets; actual results depend on dataset size and GPU runtime.*

---

## Real-World Validation

Record 5–10 bilingual speakers reading the test subset and place WAV files in `data/human_recordings/` with a matching `manifest_human.json`. Run `evaluate.py` with `--manifest_path data/manifest_human.json` to compare synthetic vs human performance.
