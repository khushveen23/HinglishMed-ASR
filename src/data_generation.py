"""
Pillar 1 & 2 — Data Generation
================================
Generates 5,000 Hinglish clinical sentences via an LLM, then synthesises
multi-speaker audio with Coqui XTTS-v2 (20 synthetic speaker identities).
Falls back to gTTS if XTTS is unavailable (e.g. CPU-only Colab free tier).

Usage (Google Colab):
    python src/data_generation.py \
        --output_dir /content/drive/MyDrive/MedASR_Masters/data \
        --n_sentences 5000 \
        --n_speakers 20 \
        --openai_api_key $OPENAI_API_KEY
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Medical vocabulary seed (Pillar 1 — LLM-driven expansion)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_NAMES = [
    "Paracetamol", "Metformin", "Amlodipine", "Atorvastatin", "Azithromycin",
    "Amoxicillin", "Omeprazole", "Pantoprazole", "Ibuprofen", "Cetirizine",
    "Aspirin", "Clopidogrel", "Losartan", "Enalapril", "Furosemide",
    "Metoprolol", "Levothyroxine", "Insulin Glargine", "Dexamethasone",
    "Ondansetron", "Diazepam", "Phenytoin", "Sumatriptan", "Hydroxychloroquine",
    "Doxycycline", "Ciprofloxacin", "Ceftriaxone", "Vancomycin", "Rifampicin",
    "Isoniazid", "Prednisolone", "Montelukast", "Salbutamol", "Budesonide",
]

SYMPTOMS = [
    "bukhar", "chest pain", "breathlessness", "BP high", "swelling",
    "loose motions", "sugar high", "cough", "back pain", "eye irritation",
    "headache", "vomiting", "dizziness", "fatigue", "joint pain",
    "skin rash", "ear pain", "stomach pain", "constipation", "acidity",
    "palpitations", "numbness", "weakness", "weight loss", "night sweats",
]

DIAGNOSES = [
    "Type 2 Diabetes", "Hypertension", "Hypothyroidism", "Dengue", "Malaria",
    "UTI", "Pneumonia", "Appendicitis", "Kidney Stone", "Migraine",
    "Anemia", "Gout", "GERD", "Asthma", "COPD", "Heart Failure",
    "Stroke", "Epilepsy", "Dementia", "Sepsis",
]

SCENARIO_TYPES = [
    "patient_history",
    "phone_consultation",
    "emergency_directive",
    "prescription_dictation",
    "lab_result_interpretation",
    "follow_up_advice",
    "pediatric_case",
    "cardiac_case",
    "surgical_case",
    "lifestyle_counseling",
]

# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven sentence generation
# ─────────────────────────────────────────────────────────────────────────────

def build_llm_prompt(scenario_type: str, drug: str, symptom: str, diagnosis: str) -> str:
    """Build a prompt that asks the LLM to generate one Hinglish clinical sentence."""
    return (
        f"You are a bilingual Indian doctor who naturally mixes Hindi and English "
        f"(Hinglish) when speaking. Generate exactly ONE realistic clinical sentence "
        f"for the following scenario.\n\n"
        f"Scenario type : {scenario_type}\n"
        f"Drug involved : {drug}\n"
        f"Symptom       : {symptom}\n"
        f"Diagnosis     : {diagnosis}\n\n"
        f"Rules:\n"
        f"- Mix Hindi (Devanagari script) and English naturally.\n"
        f"- Include dosage numbers, medical abbreviations, or test values where relevant.\n"
        f"- Keep it to 1–2 sentences, as a doctor would say it aloud.\n"
        f"- Do NOT add any explanation or prefix — output only the sentence.\n"
    )


def generate_sentences_with_llm(n: int, api_key: str, model: str = "gpt-4o-mini") -> list[str]:
    """Call OpenAI to generate n unique Hinglish clinical sentences."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")

    client = OpenAI(api_key=api_key)
    sentences: list[str] = []
    seen: set[str] = set()

    print(f"Generating {n} sentences via {model}...")
    with tqdm(total=n) as pbar:
        while len(sentences) < n:
            drug = random.choice(DRUG_NAMES)
            symptom = random.choice(SYMPTOMS)
            diagnosis = random.choice(DIAGNOSES)
            scenario = random.choice(SCENARIO_TYPES)
            prompt = build_llm_prompt(scenario, drug, symptom, diagnosis)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=120,
                )
                text = response.choices[0].message.content.strip()
                if text and text not in seen:
                    sentences.append(text)
                    seen.add(text)
                    pbar.update(1)
                time.sleep(0.1)  # rate-limit guard
            except Exception as e:
                print(f"\nLLM error: {e}. Retrying in 5s...")
                time.sleep(5)

    return sentences


def generate_sentences_fallback(n: int) -> list[str]:
    """
    Fallback: expand the 105 seed sentences from the original notebook
    using simple template substitution to reach n sentences.
    Used when no API key is provided.
    """
    SEED_SENTENCES = [
        "पेशेंट को तीन दिन से बुखार है और सर दर्द भी हो रहा है",
        "मुझे सुबह उठते ही चेस्ट पेन होती है",
        "उसकी बीपी रीडिंग 140 ओवर 90 है जो हाई है",
        "पेशेंट को ब्रीदलेसनेस हो रही है एस्पेशली रात को",
        "हाथ पैर में बहुत स्वेलिंग है पिछले दो दिन से",
        "पेट में दर्द है और लूज़ मोशन भी चल रहे हैं",
        "उसकी शुगर फास्टिंग में 180 आई है जो कंसर्निंग है",
        "पेशेंट को कफ है और बलगम भी आ रहा है",
        "बैक पेन बहुत सीवियर है और लेग तक जा रहा है",
        "आँखों में इरिटेशन और रेडनेस है कल से",
        "पैरासिटामोल 500mg सुबह शाम खाना खाने के बाद लेनी है",
        "मेटफॉर्मिन 500mg एक टैबलेट दिन में दो बार लेनी है",
        "ओमेप्राज़ोल खाने से 30 मिनट पहले लेना ज़रूरी है",
        "अम्लोडिपिन 5mg रोज़ रात को सोने से पहले लेनी है",
        "एज़िथ्रोमाइसिन 500mg 5 दिन तक कम्पलीट करनी है",
        "इबुप्रोफेन 400mg पेन के वक्त लेते हैं खाना खाने के साथ",
        "एटोरवास्टेटिन 10mg रात को सोने से पहले लेनी है",
        "अमोक्सिसिलिन 500mg तीन बार दिन में लेनी है",
        "पैंटोप्राज़ोल सुबह खाली पेट लेनी है",
        "सेटिरिज़िन 10mg एक टैबलेट रात को लेनी है एलर्जी के लिए",
        "इस पेशेंट को टाइप 2 डायबिटीज़ है और उन्हें डाइट कंट्रोल करना होगा",
        "हाइपरटेंशन का डायग्नोसिस कन्फर्म हुआ है और लाइफस्टाइल चेंजेस ज़रूरी हैं",
        "ईसीजी में साइनस टैकीकार्डिया दिख रही है",
        "चेस्ट एक्सरे नॉर्मल है कोई कंसोलिडेशन नहीं है",
        "एचबीए1सी 8.5 परसेंट आया है जो पूअर ग्लाइसेमिक कंट्रोल दिखाता है",
        "कम्पलीट ब्लड काउंट में हीमोग्लोबिन लो है एनीमिया है",
        "यूरिन कल्चर में ई कोलाई ग्रोथ आई है यूटीआई कन्फर्म है",
        "थायरॉइड टीएसएच लेवल हाई है हाइपोथायरॉइडिज़्म का केस है",
        "डेंगू एनएस1 एंटीजन पॉज़िटिव आया है एडमिट करना पड़ेगा",
        "किडनी फंक्शन टेस्ट में क्रिएटिनिन थोड़ा एलिवेटेड है",
    ]

    templates = [
        "पेशेंट को {drug} {dose}mg दिन में {n} बार लेनी है {symptom} के लिए",
        "{drug} {dose}mg {time} लेना है {diagnosis} में",
        "पेशेंट की {test} रिपोर्ट में {value} आया है {diagnosis} कन्फर्म है",
        "{symptom} की शिकायत है {drug} {dose}mg प्रिस्क्राइब किया है",
        "फॉलो अप में {diagnosis} कंट्रोल हो रहा है {drug} जारी रखें",
    ]
    doses = [250, 500, 1000, 5, 10, 20, 40, 100, 200]
    times = ["सुबह", "रात को", "खाने के बाद", "खाली पेट", "दिन में दो बार"]
    tests = ["HbA1c", "CBC", "LFT", "KFT", "TSH", "Lipid Profile", "Urine Culture"]
    values = ["8.5%", "लो", "हाई", "नॉर्मल", "एलिवेटेड", "पॉज़िटिव"]

    generated = list(SEED_SENTENCES)
    random.seed(42)
    while len(generated) < n:
        tmpl = random.choice(templates)
        s = tmpl.format(
            drug=random.choice(DRUG_NAMES),
            dose=random.choice(doses),
            n=random.randint(1, 3),
            symptom=random.choice(SYMPTOMS),
            diagnosis=random.choice(DIAGNOSES),
            time=random.choice(times),
            test=random.choice(tests),
            value=random.choice(values),
        )
        generated.append(s)
    return generated[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Multi-speaker audio synthesis (Pillar 1 — XTTS upgrade)
# ─────────────────────────────────────────────────────────────────────────────

# 20 synthetic speaker profiles: (gender, speaking_rate_modifier, description)
SPEAKER_PROFILES = [
    {"id": "spk_00", "gender": "female", "rate": 1.0,  "desc": "young_female_clear"},
    {"id": "spk_01", "gender": "male",   "rate": 0.9,  "desc": "middle_aged_male_slow"},
    {"id": "spk_02", "gender": "female", "rate": 1.1,  "desc": "young_female_fast"},
    {"id": "spk_03", "gender": "male",   "rate": 1.0,  "desc": "elderly_male_clear"},
    {"id": "spk_04", "gender": "female", "rate": 0.85, "desc": "elderly_female_slow"},
    {"id": "spk_05", "gender": "male",   "rate": 1.15, "desc": "young_male_fast"},
    {"id": "spk_06", "gender": "female", "rate": 1.0,  "desc": "middle_aged_female"},
    {"id": "spk_07", "gender": "male",   "rate": 0.95, "desc": "middle_aged_male_clear"},
    {"id": "spk_08", "gender": "female", "rate": 1.05, "desc": "young_female_medium"},
    {"id": "spk_09", "gender": "male",   "rate": 1.1,  "desc": "young_male_medium"},
    {"id": "spk_10", "gender": "female", "rate": 0.9,  "desc": "elderly_female_clear"},
    {"id": "spk_11", "gender": "male",   "rate": 1.2,  "desc": "young_male_very_fast"},
    {"id": "spk_12", "gender": "female", "rate": 0.8,  "desc": "elderly_female_very_slow"},
    {"id": "spk_13", "gender": "male",   "rate": 0.85, "desc": "elderly_male_slow"},
    {"id": "spk_14", "gender": "female", "rate": 1.15, "desc": "young_female_fast2"},
    {"id": "spk_15", "gender": "male",   "rate": 1.0,  "desc": "middle_aged_male2"},
    {"id": "spk_16", "gender": "female", "rate": 0.95, "desc": "middle_aged_female2"},
    {"id": "spk_17", "gender": "male",   "rate": 1.05, "desc": "young_male_clear"},
    {"id": "spk_18", "gender": "female", "rate": 1.0,  "desc": "young_female_neutral"},
    {"id": "spk_19", "gender": "male",   "rate": 0.9,  "desc": "middle_aged_male_slow2"},
]


def synthesise_with_xtts(
    sentences: list[str],
    audio_dir: Path,
    speaker_profiles: list[dict],
    sample_rate: int = 16000,
) -> list[dict]:
    """
    Synthesise audio using Coqui XTTS-v2.
    Each sentence is rendered by every speaker profile.
    Returns a manifest list of {audio_path, text, speaker_id, split}.
    """
    try:
        from TTS.api import TTS  # pip install TTS
        import torch
        import soundfile as sf
        import numpy as np
    except ImportError:
        raise ImportError("Run: pip install TTS soundfile")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device}...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Use built-in reference speakers from XTTS speaker list
    available_speakers = tts.speakers or []
    # Map our 20 profiles to available XTTS speakers (cycle if fewer available)
    speaker_map = {}
    for i, profile in enumerate(speaker_profiles):
        if available_speakers:
            speaker_map[profile["id"]] = available_speakers[i % len(available_speakers)]
        else:
            speaker_map[profile["id"]] = None

    manifest = []
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesising {len(sentences)} sentences × {len(speaker_profiles)} speakers...")
    for sent_idx, sentence in enumerate(tqdm(sentences, desc="Sentences")):
        for profile in speaker_profiles:
            spk_id = profile["id"]
            fname = f"sent_{sent_idx:05d}_{spk_id}.wav"
            fpath = audio_dir / fname

            if fpath.exists():
                manifest.append({"audio_path": str(fpath), "text": sentence,
                                  "speaker_id": spk_id, "split": "train"})
                continue

            try:
                xtts_speaker = speaker_map[spk_id]
                wav = tts.tts(
                    text=sentence,
                    speaker=xtts_speaker,
                    language="hi",
                    speed=profile["rate"],
                )
                import numpy as np
                wav_np = np.array(wav, dtype=np.float32)
                import soundfile as sf
                sf.write(str(fpath), wav_np, sample_rate)
                manifest.append({"audio_path": str(fpath), "text": sentence,
                                  "speaker_id": spk_id, "split": "train"})
            except Exception as e:
                print(f"\nXTTS failed for sent {sent_idx}, {spk_id}: {e}")

    return manifest


def synthesise_with_gtts_fallback(
    sentences: list[str],
    audio_dir: Path,
) -> list[dict]:
    """
    Fallback TTS using gTTS (single speaker, Hindi).
    Used when XTTS is unavailable.
    """
    from gtts import gTTS
    from pydub import AudioSegment
    import io

    manifest = []
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Synthesising {len(sentences)} sentences with gTTS (fallback)...")

    for i, sentence in enumerate(tqdm(sentences, desc="gTTS")):
        fname = f"sent_{i:05d}_spk_00.wav"
        fpath = audio_dir / fname
        if fpath.exists():
            manifest.append({"audio_path": str(fpath), "text": sentence,
                              "speaker_id": "spk_00", "split": "train"})
            continue
        try:
            tts = gTTS(text=sentence, lang="hi", slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            audio = AudioSegment.from_mp3(mp3_buf)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(str(fpath), format="wav")
            manifest.append({"audio_path": str(fpath), "text": sentence,
                              "speaker_id": "spk_00", "split": "train"})
            time.sleep(0.3)
        except Exception as e:
            print(f"\ngTTS failed for sentence {i}: {e}")
            time.sleep(2)

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Train / validation / test split
# ─────────────────────────────────────────────────────────────────────────────

def assign_splits(manifest: list[dict], train_ratio=0.8, val_ratio=0.1) -> list[dict]:
    """
    Assign train/val/test splits at the sentence level so the same sentence
    does not appear in both train and test (even across speakers).
    """
    # Group by text
    from collections import defaultdict
    by_text: dict[str, list] = defaultdict(list)
    for item in manifest:
        by_text[item["text"]].append(item)

    texts = list(by_text.keys())
    random.seed(42)
    random.shuffle(texts)

    n = len(texts)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_texts = set(texts[:n_train])
    val_texts   = set(texts[n_train:n_train + n_val])
    # rest → test

    for item in manifest:
        if item["text"] in train_texts:
            item["split"] = "train"
        elif item["text"] in val_texts:
            item["split"] = "validation"
        else:
            item["split"] = "test"

    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedASR data generation pipeline")
    parser.add_argument("--output_dir",    default="data",   help="Root output directory")
    parser.add_argument("--n_sentences",   type=int, default=5000)
    parser.add_argument("--n_speakers",    type=int, default=20)
    parser.add_argument("--openai_api_key", default=None,
                        help="OpenAI API key for LLM sentence generation")
    parser.add_argument("--use_xtts",      action="store_true",
                        help="Use Coqui XTTS-v2 (requires GPU + ~5GB VRAM)")
    parser.add_argument("--llm_model",     default="gpt-4o-mini")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir  = output_dir / "audio_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate sentences ──────────────────────────────────────────
    sentences_path = output_dir / "sentences.json"
    if sentences_path.exists():
        print(f"Loading existing sentences from {sentences_path}")
        with open(sentences_path) as f:
            sentences = json.load(f)
    elif args.openai_api_key:
        sentences = generate_sentences_with_llm(
            args.n_sentences, args.openai_api_key, args.llm_model
        )
        with open(sentences_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(sentences)} sentences → {sentences_path}")
    else:
        print("No API key provided — using template-based fallback generation.")
        sentences = generate_sentences_fallback(args.n_sentences)
        with open(sentences_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(sentences)} sentences → {sentences_path}")

    # ── Step 2: Synthesise audio ─────────────────────────────────────────────
    profiles = SPEAKER_PROFILES[:args.n_speakers]
    manifest_path = output_dir / "manifest_raw.json"

    if manifest_path.exists():
        print(f"Loading existing manifest from {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        if args.use_xtts:
            manifest = synthesise_with_xtts(sentences, audio_dir, profiles)
        else:
            print("XTTS not requested — using gTTS fallback (single speaker).")
            print("For multi-speaker synthesis, re-run with --use_xtts on a GPU runtime.")
            manifest = synthesise_with_gtts_fallback(sentences, audio_dir)

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Saved manifest ({len(manifest)} entries) → {manifest_path}")

    # ── Step 3: Assign splits ────────────────────────────────────────────────
    manifest = assign_splits(manifest)
    final_manifest_path = output_dir / "manifest.json"
    with open(final_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── Step 4: Save CSV summary ─────────────────────────────────────────────
    df = pd.DataFrame(manifest)
    csv_path = output_dir / "manifest.csv"
    df.to_csv(csv_path, index=False)

    print("\n── Dataset Summary ──────────────────────────────────────────")
    print(df["split"].value_counts().to_string())
    print(f"Total audio files : {len(manifest)}")
    print(f"Unique sentences  : {df['text'].nunique()}")
    print(f"Unique speakers   : {df['speaker_id'].nunique()}")
    print(f"Manifest CSV      : {csv_path}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
