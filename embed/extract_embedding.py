#!/usr/bin/env python
"""
Extract a Kokoro-compatible speaker embedding from processed recordings.

Strategy
--------
1. Inspect an existing Kokoro .npy voice to confirm the expected shape/dtype.
2. Try to extract style via Kokoro's own internal style encoder
   (most compatible — uses the same model the TTS runtime uses).
3. Fall back to resemblyzer (256-dim d-vector) if Kokoro extraction fails.
4. Average embeddings across all recordings.
5. Save jenny.npy and optionally copy it to the KokoroTTS voices directory.

Usage:
    python extract_embedding.py [--input recordings_processed]
                                [--output jenny.npy]
                                [--deploy C:\\Code\\Text2Speech\\voices\\jenny.npy]
                                [--no-deploy]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

VOICES_DIR   = r"C:\Code\Text2Speech\voices"
DEFAULT_DEPLOY = os.path.join(VOICES_DIR, "jenny.npy")


# ── Step 1: inspect existing voice format ────────────────────────────────────

def inspect_voices(voices_dir: str) -> dict | None:
    """Load one existing .npy voice and report its structure."""
    if not os.path.isdir(voices_dir):
        print(f"  Voices directory not found: {voices_dir}")
        return None

    npy_files = sorted(f for f in os.listdir(voices_dir) if f.endswith(".npy"))
    if not npy_files:
        print(f"  No .npy files in {voices_dir}")
        return None

    sample_path = os.path.join(voices_dir, npy_files[0])
    try:
        v = np.load(sample_path, allow_pickle=True)
        if v.dtype == object:
            item = v.item()
            if isinstance(item, dict):
                print(f"  Format ({npy_files[0]}): dict with keys {list(item.keys())}")
                for k, val in item.items():
                    if hasattr(val, "shape"):
                        print(f"    [{k}]: shape={val.shape}  dtype={val.dtype}")
                return {"type": "dict", "sample": item}
            else:
                print(f"  Format ({npy_files[0]}): object → {type(item)}")
                return {"type": "object", "sample": item}
        else:
            print(f"  Format ({npy_files[0]}): ndarray  shape={v.shape}  dtype={v.dtype}")
            return {"type": "ndarray", "shape": v.shape, "dtype": v.dtype}
    except Exception as exc:
        print(f"  Could not inspect {sample_path}: {exc}")
        return None


# ── Step 2a: extract via Kokoro's style encoder ───────────────────────────────

def extract_kokoro(wav_paths: list[str]) -> np.ndarray | None:
    """
    Use the Kokoro / StyleTTS2 style encoder.
    compute_style(path) → tensor of shape (1, 256) on Kokoro-82M.
    The first 128 dims are ref_s (style), next 128 are ref_p (prosody).
    """
    try:
        import torch
        from kokoro import KPipeline

        print("  Loading Kokoro pipeline…")
        pipe = KPipeline(lang_code="a")

        model = pipe.model
        if not hasattr(model, "compute_style"):
            print("  KPipeline.model has no compute_style — skipping Kokoro method.")
            return None

        embeddings = []
        for path in wav_paths:
            try:
                with torch.no_grad():
                    style = model.compute_style(path)        # (1, 256)
                arr = style.cpu().float().numpy()
                embeddings.append(arr)
                print(f"  OK  {os.path.basename(path):30s}  shape={arr.shape}")
            except Exception as exc:
                print(f"  ERR {os.path.basename(path)}: {exc}")

        if not embeddings:
            return None

        result = np.mean(embeddings, axis=0)
        print(f"\n  Averaged {len(embeddings)} embeddings → shape={result.shape}  dtype={result.dtype}")
        return result

    except ImportError:
        print("  kokoro not installed — skipping.")
        return None
    except Exception as exc:
        print(f"  Kokoro extraction failed: {exc}")
        return None


# ── Step 2b: fallback — resemblyzer d-vector ─────────────────────────────────

def extract_resemblyzer(wav_paths: list[str]) -> np.ndarray | None:
    """
    256-dim d-vector via resemblyzer. Compatible with many TTS systems.
    Note: if the Kokoro format check shows a different shape, you may need
    to reshape or re-train. This is a best-effort fallback.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        print("  Loading VoiceEncoder…")
        encoder = VoiceEncoder(device="cpu")

        embeddings = []
        for path in wav_paths:
            try:
                wav = preprocess_wav(path, source_sr=24_000)
                embed = encoder.embed_utterance(wav)          # (256,)
                embeddings.append(embed)
                print(f"  OK  {os.path.basename(path):30s}  shape={embed.shape}")
            except Exception as exc:
                print(f"  ERR {os.path.basename(path)}: {exc}")

        if not embeddings:
            return None

        result = np.mean(embeddings, axis=0)
        print(f"\n  Averaged {len(embeddings)} embeddings → shape={result.shape}  dtype={result.dtype}")
        return result

    except ImportError:
        print("  resemblyzer not installed — skipping.")
        return None
    except Exception as exc:
        print(f"  resemblyzer extraction failed: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Extract Kokoro speaker embedding")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(script_dir, "recordings_processed"),
        help="Directory of pre-processed WAV files (output of preprocess.py)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(script_dir, "jenny.npy"),
        help="Where to save the embedding",
    )
    parser.add_argument(
        "--deploy",
        default=DEFAULT_DEPLOY,
        help="Also copy result here (KokoroTTS voices directory)",
    )
    parser.add_argument(
        "--no-deploy", action="store_true",
        help="Skip copying to the voices directory",
    )
    args = parser.parse_args()

    input_dir   = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    # ── Find processed WAVs ──────────────────────────────────────────────────
    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory not found: {input_dir}")
        print("Run preprocess.py first.")
        sys.exit(1)

    wavs = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".wav")
    )
    if not wavs:
        print(f"No WAV files in {input_dir}")
        sys.exit(1)

    print(f"Found {len(wavs)} processed files in {input_dir}\n")

    # ── Step 1: inspect target format ───────────────────────────────────────
    print("--- Inspecting existing Kokoro voice format ---")
    fmt = inspect_voices(VOICES_DIR)
    print()

    # ── Step 2: extract embeddings ───────────────────────────────────────────
    print("--- Extracting speaker embeddings (Kokoro method) ---")
    embedding = extract_kokoro(wavs)

    if embedding is None:
        print("\n--- Kokoro method unavailable; trying resemblyzer ---")
        embedding = extract_resemblyzer(wavs)

    if embedding is None:
        print("\nERROR: All extraction methods failed.")
        print("Check that requirements are installed: pip install -r requirements.txt")
        sys.exit(1)

    # ── Step 3: save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, embedding)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {embedding.shape}  dtype: {embedding.dtype}")

    # Sanity check against detected format
    if fmt and fmt["type"] == "ndarray":
        expected = fmt["shape"]
        if embedding.shape != expected:
            print(f"\nWARNING: shape {embedding.shape} does not match existing voices {expected}.")
            print("The TTS system may reject this embedding or produce distorted audio.")
            print("Consider adjusting the extraction method or reshaping the array.")

    # ── Step 4: deploy ───────────────────────────────────────────────────────
    if not args.no_deploy:
        deploy_dir = os.path.dirname(args.deploy)
        if os.path.isdir(deploy_dir):
            shutil.copy2(output_path, args.deploy)
            print(f"Deployed to: {args.deploy}")
        else:
            print(f"\nNote: Deploy path not found: {deploy_dir}")
            print(f"Manually copy {output_path} to your KokoroTTS voices directory.")

    print("\nDone!")


if __name__ == "__main__":
    main()
