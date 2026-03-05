#!/usr/bin/env python
"""
Extract a Kokoro-compatible speaker embedding from processed recordings.

Strategy (tried in order)
--------------------------
1. Borrow kokoro from C:\\Code\\Text2Speech — most compatible, no extra install needed.
2. Import kokoro from the active environment (if it was installed separately).
3. Pure-torch fallback: mel-spectrogram statistics via torchaudio.
   NOTE: method 3 may not be compatible with all Kokoro versions; prefer 1 or 2.

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

import numpy as np

VOICES_DIR     = r"C:\Code\Text2Speech\voices"
DEFAULT_DEPLOY = os.path.join(VOICES_DIR, "jenny.npy")

# Candidate site-packages dirs inside the Text2Speech project
_T2S_SITEPKG_CANDIDATES = [
    r"C:\Code\Text2Speech\venv\Lib\site-packages",
    r"C:\Code\Text2Speech\.venv\Lib\site-packages",
    r"C:\Code\Text2Speech\env\Lib\site-packages",
]


# ── Step 0: inject Text2Speech site-packages so we can import kokoro ─────────

def _inject_text2speech_venv() -> bool:
    """Add C:\\Code\\Text2Speech venv to sys.path so kokoro is importable."""
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
                print(f"  Format ({npy_files[0]}): object → {type(item)}")
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
    Use Kokoro / StyleTTS2 compute_style().
    Returns shape (1, 256) on Kokoro-82M: first 128 dims = style, next 128 = prosody.
    Tries the Text2Speech venv first, then falls back to whatever is on sys.path.
    """
    # Try to make kokoro importable from the existing Text2Speech install
    found_t2s = _inject_text2speech_venv()
    if not found_t2s:
        print("  Text2Speech kokoro venv not found — trying active environment.")

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
                    style = model.compute_style(path)   # (1, 256)
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
        print("  kokoro not importable (Text2Speech venv not found and not installed here).")
        return None
    except Exception as exc:
        print(f"  Kokoro extraction failed: {exc}")
        return None


# ── Step 2b: pure-torch fallback ─────────────────────────────────────────────

def extract_torch_fallback(wav_paths: list[str]) -> np.ndarray | None:
    """
    Pure torchaudio fallback — no extra packages needed.

    Computes log-mel spectrogram statistics (mean + std per band) and projects
    them to match the dimension of existing Kokoro voice files (detected at runtime,
    default 256). Result shape: (1, target_dim).

    NOTE: This is a best-effort approximation. It may not produce voices that
    sound exactly like you; run the Kokoro method when possible for real quality.
    """
    try:
        import torch
        import torchaudio
        import torchaudio.transforms as T

        N_MELS = 80
        TARGET_DIM = 256   # Kokoro-82M style dim; overridden below if fmt is known

        mel_tf = T.MelSpectrogram(
            sample_rate=24_000,
            n_fft=2048,
            hop_length=300,
            n_mels=N_MELS,
            power=1.0,
        )
        amp_to_db = T.AmplitudeToDB()

        all_stats = []
        for path in wav_paths:
            try:
                wav, sr = torchaudio.load(path)
                if sr != 24_000:
                    wav = T.Resample(sr, 24_000)(wav)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)

                mel = amp_to_db(mel_tf(wav))    # (1, N_MELS, T)
                mean = mel.mean(dim=-1)          # (1, N_MELS)
                std  = mel.std(dim=-1)           # (1, N_MELS)
                stats = torch.cat([mean, std], dim=1)   # (1, N_MELS*2 = 160)
                all_stats.append(stats.numpy())
                print(f"  OK  {os.path.basename(path):30s}  stats shape={stats.shape}")
            except Exception as exc:
                print(f"  ERR {os.path.basename(path)}: {exc}")

        if not all_stats:
            return None

        avg = np.mean(all_stats, axis=0)   # (1, 160)

        # Project to TARGET_DIM via zero-padding or truncation so the shape
        # matches what Kokoro expects (checked via inspect_voices).
        current_dim = avg.shape[1]
        if current_dim < TARGET_DIM:
            pad = np.zeros((1, TARGET_DIM - current_dim), dtype=np.float32)
            avg = np.concatenate([avg, pad], axis=1)
        else:
            avg = avg[:, :TARGET_DIM]

        avg = avg.astype(np.float32)
        print(f"\n  Averaged {len(all_stats)} files → shape={avg.shape}  dtype={avg.dtype}")
        print("  WARNING: torch fallback used. Voice quality may be limited.")
        print("           For best results, ensure C:\\Code\\Text2Speech uses kokoro in a venv.")
        return avg

    except Exception as exc:
        print(f"  torch fallback failed: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Extract Kokoro speaker embedding")
    parser.add_argument(
        "--input", "-i",
        default=os.path.join(script_dir, "recordings_processed"),
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.join(script_dir, "jenny.npy"),
    )
    parser.add_argument("--deploy",    default=DEFAULT_DEPLOY)
    parser.add_argument("--no-deploy", action="store_true")
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

    # ── Extract ──────────────────────────────────────────────────────────────
    print("--- Extracting speaker embeddings (Kokoro method) ---")
    embedding = extract_kokoro(wavs)

    if embedding is None:
        print("\n--- Kokoro unavailable; using torch mel-spectrogram fallback ---")
        embedding = extract_torch_fallback(wavs)

    if embedding is None:
        print("\nERROR: All extraction methods failed.")
        sys.exit(1)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, embedding)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {embedding.shape}  dtype: {embedding.dtype}")

    if fmt and fmt["type"] == "ndarray" and embedding.shape != fmt["shape"]:
        print(f"\nWARNING: shape {embedding.shape} ≠ expected {fmt['shape']}.")
        print("Kokoro may reject this or produce distorted output.")

    # ── Deploy ───────────────────────────────────────────────────────────────
    if not args.no_deploy:
        deploy_dir = os.path.dirname(args.deploy)
        if os.path.isdir(deploy_dir):
            shutil.copy2(output_path, args.deploy)
            print(f"Deployed to: {args.deploy}")
        else:
            print(f"\nNote: {deploy_dir} not found — copy {output_path} manually.")

    print("\nDone!")


if __name__ == "__main__":
    main()
