# HinglishMed-ASR — Hinglish Clinical Speech Recognition

A research-grade Automatic Speech Recognition (ASR) system for **Hinglish (Hindi-English code-switched) clinical speech**, built on OpenAI Whisper with improvements across data generation, model adaptation, clinical evaluation, hybrid decoding, and MLOps.

The goal is to improve transcription of the way Indian doctors and patients naturally communicate, where Hindi and English are frequently mixed within the same sentence.

---

## The Problem

In Indian healthcare settings, conversations often contain **Hindi-English code-switching** along with medical terminology:

> "उसकी BP बहुत high है, Metformin 500mg दे रहा हूँ"

General-purpose speech recognition systems can struggle with:

* Hindi-English code-switching
* Medical terminology and drug names
* Dosages and laboratory values
* Indian-accented speech
* Noisy clinical environments
* Spoken medical terms that require normalization

HinglishMed-ASR explores domain adaptation of Whisper specifically for this underrepresented clinical speech setting.

---

## What This Project Does

HinglishMed-ASR implements an end-to-end pipeline:

1. **Clinical corpus generation**
2. **Multi-speaker speech synthesis**
3. **Acoustic data augmentation**
4. **Whisper fine-tuning**
5. **Medical vocabulary adaptation**
6. **Clinical-aware evaluation**
7. **Post-processing and text normalization**
8. **Experiment tracking and model comparison**

---

## Project Structure

```text
HinglishMed-ASR/
├── src/
│   ├── data_generation.py      # Clinical dataset + multi-speaker TTS
│   ├── audio_augmentation.py   # Acoustic augmentation pipeline
│   ├── train.py                # Tokenizer expansion + LoRA/QLoRA
│   └── evaluate.py             # Evaluation, ITN, LM re-ranking, W&B
├── HinglishMed-ASR_Colab.ipynb # End-to-end Google Colab notebook
├── requirements.txt
└── README.md
```

---

# Five Research Pillars

## Pillar 1 — Scale & Realism of Data

### LLM-driven clinical corpus

Generate **5,000 unique Hinglish clinical sentences** using GPT-4o-mini, guided by a curated medical vocabulary containing:

* Drug names
* Symptoms
* Diagnoses
* Dosages
* Laboratory values
* Clinical measurements
* Medical instructions

### Multi-speaker speech synthesis

Generate speech using **Coqui XTTS-v2** across multiple synthetic speaker identities to introduce variation in:

* Gender
* Age
* Speaking rate
* Voice characteristics

### Acoustic augmentation

The training data is augmented to simulate realistic clinical environments using:

* Gaussian noise
* Telephone filtering
* Room acoustics
* Time stretching
* Pitch shifting

---

# Pillar 2 — Advanced Whisper Adaptation

## Medical tokenizer expansion

Extend the Whisper vocabulary with frequently occurring medical-Hinglish terms such as:

```text
500mg
HbA1c
BP
Metformin
ECG
140/90
```

This aims to reduce unnecessary token fragmentation for domain-specific terminology.

## LoRA Fine-tuning

Fine-tune **Whisper-medium** using Low-Rank Adaptation (LoRA), reducing the number of trainable parameters while maintaining efficient training.

Configuration:

```text
Model: Whisper-medium
Method: LoRA
Rank (r): 32
```

## QLoRA

Use 4-bit quantization to support training on memory-constrained hardware such as Google Colab GPUs.

## Prompt Conditioning

Use a medical transcription prompt to guide the decoder toward the expected domain and language style:

```text
The following is a medical transcription in Hinglish...
```

---

# Pillar 3 — Clinical Evaluation

The system evaluates transcription quality using multiple metrics.

### Word Error Rate (WER)

Standard ASR metric measuring the difference between reference and predicted transcripts.

### Clinical WER (cWER)

Measures recognition accuracy specifically for important clinical entities such as:

* Drug names
* Dosages
* Laboratory values
* Blood pressure readings
* Medical terminology

### Semantic Metrics

Additional metrics are used to measure textual similarity and preservation of clinical meaning:

* BLEU-4
* ROUGE-L

---

# Pillar 4 — Post-Processing & Hybrid Decoding

## Inverse Text Normalization

Convert spoken medical expressions into standard clinical notation.

Example:

```text
एक सो चालीस ओवर नब्बे
        ↓
140/90
```

and:

```text
500 एमजी
   ↓
500mg
```

## Language Model Re-ranking

Multiple beam-search hypotheses can be scored using **AI4Bharat Indic-BERT** pseudo-perplexity.

The system then selects the hypothesis with the strongest language-model score.

This provides an additional layer of domain-aware decoding after Whisper transcription.

---

# Pillar 5 — Production Engineering & MLOps

### Modular architecture

The project is organized into independent components:

```text
Data Generation
      ↓
Audio Augmentation
      ↓
Whisper Fine-tuning
      ↓
ASR Evaluation
      ↓
Post-processing
      ↓
Experiment Tracking
```

### Weights & Biases

W&B is used for experiment tracking, including:

* Training loss
* Validation WER
* Checkpoint comparisons
* Audio artifacts
* Evaluation metrics
* Model comparisons

### Experiment Comparison

The evaluation pipeline supports comparison between:

```text
Whisper baseline
       ↓
Full fine-tuning
       ↓
LoRA fine-tuning
       ↓
LoRA + ITN + LM re-ranking
```

---

# Tech Stack

### Programming

```text
Python
PyTorch
Pandas
Matplotlib
```

### Speech & AI

```text
OpenAI Whisper
GPT-4o-mini
Coqui XTTS-v2
gTTS
```

### Fine-tuning

```text
HuggingFace Transformers
PEFT
LoRA
QLoRA
```

### NLP & Evaluation

```text
AI4Bharat Indic-BERT
jiwer
BLEU
ROUGE
```

### Audio Processing

```text
audiomentations
```

### MLOps

```text
Weights & Biases
```

---

# Quick Start — Google Colab

Open:

```text
HinglishMed-ASR_Colab.ipynb
```

in Google Colab and select a **T4 GPU** runtime.

The notebook walks through the complete pipeline from dataset generation to model evaluation.

---

# Manual Pipeline

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Generate Clinical Dataset and Speech

```bash
python src/data_generation.py \
    --output_dir data \
    --n_sentences 5000 \
    --openai_api_key YOUR_KEY \
    --use_xtts
```

This generates the clinical sentences and synthesizes speech using multiple speaker identities.

---

## 3. Augment Audio

```bash
python src/audio_augmentation.py \
    --manifest_path data/manifest.json \
    --output_dir data/audio_augmented \
    --augment_train_only
```

---

## 4. Fine-tune Whisper with LoRA

```bash
python src/train.py \
    --manifest_path data/manifest_augmented.json \
    --output_dir models/lora_medium \
    --mode lora \
    --model_name openai/whisper-medium \
    --expand_vocab \
    --use_prompt \
    --wandb_project HinglishMed-ASR
```

---

## 5. Evaluate the Model

```bash
python src/evaluate.py \
    --manifest_path data/manifest.json \
    --model_dir models/lora_medium/best_model \
    --baseline_model openai/whisper-small \
    --output_dir results \
    --use_itn \
    --wandb_project HinglishMed-ASR
```

---

# Experiment Comparison

| Model                          | WER ↓ | cWER ↓ | BLEU-4 ↑ | ROUGE-L ↑ |
| ------------------------------ | ----: | -----: | -------: | --------: |
| Whisper-small (Baseline)       |  ~45% |   ~62% |      ~18 |       ~42 |
| Full Fine-tune (Whisper-small) |  ~22% |   ~28% |      ~51 |       ~68 |
| LoRA (Whisper-medium)          |  ~14% |   ~17% |      ~64 |       ~79 |
| LoRA + ITN + LM Re-ranking     |  ~11% |   ~13% |      ~68 |       ~83 |

> **Note:** These values are indicative targets rather than guaranteed results. Actual performance depends on the generated dataset, training configuration, hardware, and evaluation split. Replace these values with measured results once experiments are completed.

---

# Real-World Validation

Synthetic speech does not fully represent real clinical conversations.

To evaluate real-world performance, record **5–10 bilingual speakers** reading a held-out test subset and place the WAV files in:

```text
data/human_recordings/
```

Create a corresponding:

```text
manifest_human.json
```

Then run:

```bash
python src/evaluate.py \
    --manifest_path data/manifest_human.json \
    --model_dir models/lora_medium/best_model \
    --output_dir results/human
```

This allows comparison between:

```text
Synthetic Speech
       ↓
Human Speech
```

and helps measure how well the model generalizes beyond synthetic training data.

---

# Limitations

This project is a research prototype rather than a production medical transcription system.

### Synthetic speech

Training data is generated using TTS, so the model may not fully generalize to real doctor-patient conversations.

### Limited speaker diversity

Synthetic speakers cannot completely reproduce the variability of real Indian speakers, accents, pronunciation, and speaking styles.

### Dataset size

Even 5,000 generated sentences are substantially smaller than datasets used by large commercial ASR systems.

### Medical safety

The system is designed for **speech recognition and transcription**, not medical diagnosis or clinical decision-making.

Predictions should therefore not be treated as a substitute for professional medical review.

### Evaluation

Performance metrics should be interpreted within the characteristics of the generated dataset and evaluation split.

---

# Future Work

* Collect real, anonymized, consented Hinglish clinical speech
* Increase speaker and accent diversity
* Expand the medical-Hinglish vocabulary
* Improve tokenizer adaptation
* Add BART-based semantic correction
* Improve clinical entity extraction
* Evaluate on larger external datasets
* Develop real-time streaming ASR
* Optimize inference for edge and low-resource devices
* Add stronger clinical terminology validation
* Compare against additional multilingual ASR models

---

# Research Motivation

HinglishMed-ASR explores how **domain adaptation, parameter-efficient fine-tuning, vocabulary expansion, and clinical-aware post-processing** can improve speech recognition for Hindi-English code-mixed medical communication.

The project focuses on an underserved combination of:

```text
Hinglish
   +
Medical Speech
   +
Automatic Speech Recognition
```

with the goal of building a reproducible research pipeline that can eventually be evaluated on real-world clinical speech.

---

# References

The project builds upon research and open-source technologies in:

* OpenAI Whisper
* HuggingFace Transformers
* LoRA / Parameter-Efficient Fine-Tuning
* Coqui XTTS-v2
* AI4Bharat Indic-BERT
* United-MedASR and related medical ASR research

---

# Author

**Khushveen Sadiora**

HinglishMed-ASR — Hinglish Clinical Speech Recognition
