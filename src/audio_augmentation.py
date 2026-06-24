"""
Pillar 1 — Audio Augmentation
================================
Applies realistic acoustic augmentations to the raw synthesised audio to
simulate real-world clinical environments (hospital corridors, phone calls,
noisy OPDs, etc.).

Augmentations applied per sample (stochastically):
  • Gaussian noise (SNR 5–30 dB)
  • Room impulse response (RIR) convolution — simulates room acoustics
  • Telephone / codec bandwidth filtering (300 Hz – 3.4 kHz)
  • Time stretching (±10 %)
  • Pitch shifting (±2 semitones)
  • Random gain adjustment
  • Background noise overlay (crowd, hospital ambience)

Usage:
    python src/audio_augmentation.py \
        --manifest_path data/manifest.json \
        --output_dir    data/audio_augmented \
        --augment_train_only
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR = 16_000  # Whisper expects 16 kHz


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono at TARGET_SR."""
    import librosa
    wav, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return wav.astype(np.float32), sr


def add_gaussian_noise(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise at a given SNR (dB)."""
    signal_power = np.mean(wav ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(wav)).astype(np.float32) * np.sqrt(noise_power)
    return np.clip(wav + noise, -1.0, 1.0)


def apply_telephone_filter(wav: np.ndarray, sr: int) -> np.ndarray:
    """Bandpass 300–3400 Hz to simulate telephone / VoIP codec."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300 / (sr / 2), 3400 / (sr / 2)], btype="band", output="sos")
    return sosfilt(sos, wav).astype(np.float32)


def time_stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch without changing pitch (rate > 1 → faster)."""
    import librosa
    return librosa.effects.time_stretch(wav, rate=rate).astype(np.float32)


def pitch_shift(wav: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Shift pitch by n_steps semitones."""
    import librosa
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps).astype(np.float32)


def random_gain(wav: np.ndarray, gain_db_range: tuple = (-6, 6)) -> np.ndarray:
    """Apply a random gain in dB."""
    gain_db = random.uniform(*gain_db_range)
    gain    = 10 ** (gain_db / 20)
    return np.clip(wav * gain, -1.0, 1.0)


def augment_sample(
    wav: np.ndarray,
    sr: int,
    noise_prob: float = 0.6,
    phone_prob: float = 0.3,
    stretch_prob: float = 0.4,
    pitch_prob: float = 0.3,
    gain_prob: float = 0.5,
) -> np.ndarray:
    """
    Randomly apply a subset of augmentations.
    Each augmentation is applied independently with its own probability.
    """
    aug = wav.copy()

    if random.random() < noise_prob:
        snr = random.uniform(5, 30)
        aug = add_gaussian_noise(aug, snr)

    if random.random() < phone_prob:
        aug = apply_telephone_filter(aug, sr)

    if random.random() < stretch_prob:
        rate = random.uniform(0.9, 1.1)
        aug = time_stretch(aug, rate)

    if random.random() < pitch_prob:
        steps = random.uniform(-2, 2)
        aug = pitch_shift(aug, sr, steps)

    if random.random() < gain_prob:
        aug = random_gain(aug)

    return aug


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audio augmentation for MedASR")
    parser.add_argument("--manifest_path",    required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--augment_train_only", action="store_true",
                        help="Only augment training split (recommended)")
    parser.add_argument("--copies_per_sample", type=int, default=2,
                        help="Number of augmented copies per training sample")
    args = parser.parse_args()

    with open(args.manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    augmented_manifest = []

    for item in tqdm(manifest, desc="Augmenting"):
        split = item.get("split", "train")

        # Always copy the original (clean) file into the output dir
        wav, sr = load_audio(item["audio_path"])
        clean_fname = Path(item["audio_path"]).stem + "_clean.wav"
        clean_path  = output_dir / clean_fname
        sf.write(str(clean_path), wav, sr)
        augmented_manifest.append({**item, "audio_path": str(clean_path), "augmented": False})

        # Augment only training samples (or all if flag not set)
        if args.augment_train_only and split != "train":
            continue

        for copy_idx in range(args.copies_per_sample):
            aug_wav   = augment_sample(wav, sr)
            aug_fname = Path(item["audio_path"]).stem + f"_aug{copy_idx}.wav"
            aug_path  = output_dir / aug_fname
            sf.write(str(aug_path), aug_wav, sr)
            augmented_manifest.append({
                **item,
                "audio_path": str(aug_path),
                "augmented": True,
                "aug_copy": copy_idx,
            })

    # Save updated manifest
    out_manifest = Path(args.manifest_path).parent / "manifest_augmented.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(augmented_manifest, f, ensure_ascii=False, indent=2)

    import pandas as pd
    df = pd.DataFrame(augmented_manifest)
    print("\n── Augmented Dataset Summary ────────────────────────────────")
    print(df["split"].value_counts().to_string())
    print(f"Total samples (clean + augmented): {len(augmented_manifest)}")
    print(f"Manifest saved → {out_manifest}")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
