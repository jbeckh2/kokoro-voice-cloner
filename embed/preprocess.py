#!/usr/bin/env python
"""
Preprocess voice recordings for speaker embedding extraction.

  1. Resample to 24 kHz mono
  2. Stationary noise reduction (noisereduce)
  3. Trim leading / trailing silence
  4. Peak-normalize to -3 dBFS

Usage (standalone):
    python preprocess.py --input <recordings_dir> [--output <output_dir>]

The C# app calls this automatically, passing --input <path_to_recordings>.
Output goes to <output_dir> which defaults to recordings_processed/ next to
this script file.
"""

import argparse
import os
import sys

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf

TARGET_SR   = 24_000
TRIM_TOP_DB = 30        # silence threshold in dB below peak
TARGET_DBFS = -3.0      # peak normalisation target


def normalize_peak(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-7:
        return audio
    target_amp = 10 ** (target_dbfs / 20.0)
    return audio * (target_amp / peak)


def process_file(src: str, dst: str) -> bool:
    """Process a single WAV file. Returns True on success."""
    try:
        audio, sr = librosa.load(src, sr=TARGET_SR, mono=True)

        if len(audio) < TARGET_SR * 0.3:
            print(f"  SKIP {os.path.basename(src)}: too short ({len(audio)/TARGET_SR:.2f}s)")
            return False

        # Stationary noise reduction — no reference clip needed
        audio = nr.reduce_noise(y=audio, sr=TARGET_SR, stationary=True, prop_decrease=0.75)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)

        if len(audio) < TARGET_SR * 0.2:
            print(f"  SKIP {os.path.basename(src)}: too short after trim")
            return False

        # Normalise
        audio = normalize_peak(audio, TARGET_DBFS)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, audio, TARGET_SR, subtype="PCM_16")

        duration = len(audio) / TARGET_SR
        print(f"  OK  {os.path.basename(src):30s}  {duration:.2f}s")
        return True

    except Exception as exc:
        print(f"  ERR {os.path.basename(src)}: {exc}")
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Preprocess voice recordings")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(script_dir, "..", "VoiceCapture", "recordings"),
        help="Directory containing raw WAV recordings",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: recordings_processed/ next to this script)",
    )
    args = parser.parse_args()

    input_dir  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output or os.path.join(script_dir, "recordings_processed"))

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Record some audio with the VoiceCapture app first.")
        sys.exit(1)

    wavs = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".wav"))
    if not wavs:
        print(f"No WAV files found in {input_dir}")
        sys.exit(1)

    print(f"Input:  {input_dir}  ({len(wavs)} files)")
    print(f"Output: {output_dir}")
    print()

    ok = sum(
        process_file(os.path.join(input_dir, w), os.path.join(output_dir, w))
        for w in wavs
    )

    print(f"\nDone: {ok}/{len(wavs)} files written to {output_dir}")


if __name__ == "__main__":
    main()
