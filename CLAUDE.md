# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Goal

Create a custom Kokoro-compatible voice (.npy speaker embedding) from recorded speech samples.
Output voice file goes into `C:\Code\Text2Speech\voices\` for use with KokoroTTS.

## Build Commands

```bash
# VoiceCapture C# app
cd VoiceCapture
dotnet build -c Release
dotnet run

# Python embedding pipeline
cd embed
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
python preprocess.py
python extract_embedding.py
```

## Project Structure

```
VoiceClone/
  VoiceCapture/           # C# WinForms recording app
    VoiceCapture.csproj
    MainForm.cs           # Script display + NAudio recording/playback
    script.txt            # ~100 phoneme-balanced sentences (one per line)
    recordings/           # Output: line_001.wav, line_002.wav, ...
  embed/                  # Python speaker embedding pipeline
    requirements.txt
    preprocess.py         # Resample → 24kHz mono, trim silence, peak-normalize to -3dBFS
    extract_embedding.py  # Speaker encoder → average embeddings → jenny.npy
    recordings_processed/ # Output of preprocess.py
```

## Architecture

**Pipeline**: Record (C# WinForms + NAudio) → Preprocess (Python/librosa) → Extract embedding (StyleTTS2 or resemblyzer) → Deploy `.npy` to KokoroTTS voices directory.

**Recording app** (`VoiceCapture/MainForm.cs`): WinForms single window. Record toggle, playback last take, Accept & Next, Previous. Output format: 24000 Hz, mono, 16-bit PCM WAV.

**Preprocess** (`embed/preprocess.py`): Reads from `recordings/`, writes to `recordings_processed/`. Uses librosa for resampling and silence trimming.

**Embedding extraction** (`embed/extract_embedding.py`): Loads all processed WAVs, runs speaker encoder on each, averages embeddings, saves as `jenny.npy`. Before implementing, verify expected shape by loading an existing Kokoro voice:
```python
import numpy as np
v = np.load('C:/Code/Text2Speech/voices/af_heart.npy', allow_pickle=True)
print(v.shape, v.dtype)
```
Then choose the encoder that produces matching output shape (StyleTTS2 encoder preferred; `resemblyzer` as fallback).

## Key Constraints

- Recording format must be **24000 Hz, mono, 16-bit PCM WAV** throughout
- AMD GPU (RX 9600 XT) — Python scripts run on **CPU** (no CUDA)
- NAudio is the only approved audio library for the C# app
- `script.txt` should contain ~100 CMU ARCTIC-style phoneme-balanced sentences
