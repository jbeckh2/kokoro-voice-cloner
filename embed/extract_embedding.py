# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Extract a Kokoro-compatible speaker embedding from processed recordings.

Strategy (tried in order)
--------------------------
1. Borrow kokoro from C:\\code\\Text2Speech venv -- most compatible, no extra install.
2. Import kokoro from the active environment (if installed separately).
3. Librosa mel-spectrogram fallback -- no torch needed, pure numpy/librosa.

The raw (1, 256) embedding is then tiled to the full voicepack shape
(n_tokens, 1, 256) so Kokoro can load it like any other voice file.

Usage:
    python extract_embedding.py [--input recordings_processed]
                                [--output jenny.npy]
                                [--deploy <path>]
                                [--no-deploy]
"""

import argparse
import os
import shutil
import sys

import numpy as np

VOICES_DIR     = r"C:\code\Text2Speech\bin\Release\net8.0\voices"
DEFAULT_DEPLOY = os.path.join(VOICES_DIR, "jenny.npy")

_T2S_SITEPKG_CANDIDATES = [
    r"C:\code\Text2Speech\venv\Lib\site-packages",
    r"C:\code\Text2Speech\.venv\Lib\site-packages",
    r"C:\Code\Text2Speech\venv\Lib\site-packages",
    r"C:\Code\Text2Speech\bin\Release\net8.0\venv\Lib\site-packages",
]


# ── Inject Text2Speech venv so kokoro is importable ───────────────────────────

def _inject_text2speech_venv() -> bool:
    for p in _T2S_SITEPKG_CANDIDATES:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "kokoro")):
            if p not in sys.path:
                sys.path.insert(0, p)
                print(f"  Found kokoro in Text2Speech venv: {p}")
            return True
    return False


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
                print(f"  Format ({npy_files[0]}): dict  keys={list(item.keys())}")
                for k, val in item.items():
                    if hasattr(val, "shape"):
                        print(f"    [{k}]: shape={val.shape}  dtype={val.dtype}")
                return {"type": "dict", "sample": item}
            else:
                print(f"  Format ({npy_files[0]}): object -> {type(item)}")
                return {"type": "object", "sample": item}
        else:
            print(f"  Format ({npy_files[0]}): ndarray  shape={v.shape}  dtype={v.dtype}")
            return {"type": "ndarray", "shape": v.shape, "dtype": v.dtype}
    except Exception as exc:
        print(f"  Could not inspect {sample_path}: {exc}")
        return None


# ── Step 2a: extract via Kokoro's own style encoder ──────────────────────────

def extract_kokoro(wav_paths: list[str]) -> np.ndarray | None:
    """
    Use Kokoro / StyleTTS2 compute_style() -> (1, 256).
    Tries the Text2Speech venv first, then the active environment.
    """
    found_t2s = _inject_text2speech_venv()
    if not found_t2s:
        print("  Text2Speech kokoro venv not found -- trying active environment.")

    try:
        import torch
        from kokoro import KPipeline

        print("  Loading Kokoro pipeline...")
        pipe = KPipeline(lang_code="a")

        model = pipe.model
        if not hasattr(model, "compute_style"):
            print("  KPipeline.model has no compute_style -- skipping Kokoro method.")
            return None

        embeddings = []
        for path in wav_paths:
            try:
                with torch.no_grad():
                    style = model.compute_style(path)   # (1, 256)
                arr = style.cpu().float().numpy()
                embeddings.append(arr)
                print(f"  OK  {os.path.basename(path):30s}  shape={arr.shape}")
            except Exception as exc:
                print(f"  ERR {os.path.basename(path)}: {exc}")

        if not embeddings:
            return None

        result = np.mean(embeddings, axis=0)
        print(f"\n  Averaged {len(embeddings)} embeddings -> shape={result.shape}  dtype={result.dtype}")
        return result

    except ImportError:
        print("  kokoro not importable (Text2Speech venv not found and not installed here).")
        return None
    except Exception as exc:
        print(f"  Kokoro extraction failed: {exc}")
        return None


# ── Step 2b: librosa fallback ─────────────────────────────────────────────────

def extract_librosa_fallback(wav_paths: list[str]) -> np.ndarray | None:
    """
    Librosa mel-spectrogram statistics fallback.
    Uses only librosa + numpy -- no torch or C extensions required.
    Produces (1, 256) by computing mean+std of log-mel bands then padding.

    NOTE: Best-effort speaker approximation. Use the Kokoro method for quality.
    """
    try:
        import librosa

        N_MELS     = 80
        TARGET_DIM = 256

        all_stats = []
        for path in wav_paths:
            try:
                audio, sr = librosa.load(path, sr=24_000, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_fft=2048, hop_length=300, n_mels=N_MELS
                )
                mel_db = librosa.power_to_db(mel)          # (N_MELS, T)
                stats  = np.concatenate([
                    mel_db.mean(axis=1),                   # (N_MELS,)
                    mel_db.std(axis=1),                    # (N_MELS,)  -> total 160
                ])
                all_stats.append(stats)
                print(f"  OK  {os.path.basename(path):30s}")
            except Exception as exc:
                print(f"  ERR {os.path.basename(path)}: {exc}")

        if not all_stats:
            return None

        avg = np.mean(all_stats, axis=0).reshape(1, -1)   # (1, 160)

        # Pad to TARGET_DIM
        if avg.shape[1] < TARGET_DIM:
            avg = np.concatenate(
                [avg, np.zeros((1, TARGET_DIM - avg.shape[1]), dtype=np.float32)], axis=1
            )
        else:
            avg = avg[:, :TARGET_DIM]

        avg = avg.astype(np.float32)
        print(f"\n  Averaged {len(all_stats)} files -> shape={avg.shape}  dtype={avg.dtype}")
        print("  WARNING: librosa fallback used. Voice quality may be limited.")
        print("           For best results, ensure kokoro is accessible from the Text2Speech venv.")
        return avg

    except Exception as exc:
        print(f"  librosa fallback failed: {exc}")
        return None


# ── Step 3: tile to full voicepack shape ──────────────────────────────────────

def reshape_to_voicepack(embedding: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Tile a single (1, style_dim) embedding to (n_tokens, 1, style_dim).

    Kokoro voices are shape (n_tokens, 1, style_dim), e.g. (510, 1, 256).
    Repeating the same style vector across all token positions is standard for
    single-speaker voice files -- every phoneme gets the same speaker identity.
    """
    emb2d     = embedding.reshape(1, -1)      # (1, style_dim)
    n_tokens  = target_shape[0]
    n_mid     = len(target_shape) - 2         # middle dims between n_tokens and style_dim

    tiled = np.tile(emb2d, (n_tokens, 1))    # (n_tokens, style_dim)
    for _ in range(n_mid):
        tiled = tiled[:, np.newaxis, :]       # (n_tokens, 1, style_dim)

    return tiled.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Extract Kokoro speaker embedding")
    parser.add_argument("--input",  "-i", default=os.path.join(script_dir, "recordings_processed"))
    parser.add_argument("--output", "-o", default=os.path.join(script_dir, "jenny.npy"))
    parser.add_argument("--deploy",       default=DEFAULT_DEPLOY)
    parser.add_argument("--no-deploy",    action="store_true")
    args = parser.parse_args()

    input_dir   = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

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

    # ── Inspect target format ────────────────────────────────────────────────
    print("--- Inspecting existing Kokoro voice format ---")
    fmt = inspect_voices(VOICES_DIR)
    print()

    # ── Extract raw speaker embedding ────────────────────────────────────────
    print("--- Extracting speaker embeddings (Kokoro method) ---")
    embedding = extract_kokoro(wavs)

    if embedding is None:
        print("\n--- Kokoro unavailable; using librosa fallback ---")
        embedding = extract_librosa_fallback(wavs)

    if embedding is None:
        print("\nERROR: All extraction methods failed.")
        sys.exit(1)

    # ── Tile to voicepack shape ───────────────────────────────────────────────
    target_shape = (510, 1, 256)   # Kokoro-82M default
    if fmt and fmt.get("type") == "ndarray":
        target_shape = fmt["shape"]
        print(f"Target shape (from existing voices): {target_shape}")
    else:
        print(f"Target shape (default Kokoro-82M):   {target_shape}")

    voicepack = reshape_to_voicepack(embedding, target_shape)
    print(f"Voicepack: {voicepack.shape}  dtype: {voicepack.dtype}\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, voicepack)
    print(f"Saved: {output_path}")

    # ── Deploy ───────────────────────────────────────────────────────────────
    if not args.no_deploy:
        deploy_dir = os.path.dirname(args.deploy)
        if os.path.isdir(deploy_dir):
            shutil.copy2(output_path, args.deploy)
            print(f"Deployed to: {args.deploy}")
        else:
            print(f"\nNote: {deploy_dir} not found -- copy {output_path} manually.")

    print("\nDone!")


if __name__ == "__main__":
    main()
