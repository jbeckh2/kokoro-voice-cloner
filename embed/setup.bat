@echo off
:: Sets up the embed/ venv with CPU-only torch and audio packages.
:: Deliberately does NOT install kokoro or resemblyzer — they pull in spacy/thinc/blis
:: which fail to build on Python 3.13+. extract_embedding.py borrows kokoro from your
:: existing C:\Code\Text2Speech installation instead.
::
:: Run once from the embed\ directory: setup.bat

cd /d "%~dp0"

:: ── Pick a Python version with complete ML wheel coverage ─────────────────────
:: Python 3.13+ often lacks binary wheels for torch, torchaudio, etc.
:: Prefer 3.12, then 3.11, then 3.10. Fall back to default `python` with a warning.

set FOUND_PY=0

for %%v in (3.12 3.11 3.10) do (
    if !FOUND_PY!==0 (
        py -%%v --version >nul 2>&1
        if not errorlevel 1 (
            echo Using Python %%v
            py -%%v -m venv venv
            set FOUND_PY=1
        )
    )
)

if !FOUND_PY!==0 (
    echo.
    echo WARNING: Python 3.10-3.12 not found. Falling back to default "python".
    echo          If you are on Python 3.13+, some packages may fail to install.
    echo          Install Python 3.12 from python.org for best compatibility.
    echo.
    python -m venv venv
)

call venv\Scripts\activate.bat

:: ── Upgrade pip ───────────────────────────────────────────────────────────────
python -m pip install --upgrade pip

:: ── CPU-only PyTorch ─────────────────────────────────────────────────────────
:: Avoids the 2 GB CUDA download; speaker encoding runs fine on CPU.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: ── Audio processing ─────────────────────────────────────────────────────────
pip install librosa soundfile noisereduce numpy

echo.
echo Setup complete.
echo Activate with:  venv\Scripts\activate
echo.
echo NOTE: kokoro is NOT installed here on purpose.
echo       extract_embedding.py will find it in C:\Code\Text2Speech automatically.
echo       If that fails, it falls back to a torch-based mel-spectrogram approach.
