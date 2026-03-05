@echo off
:: Creates a venv and installs all dependencies (CPU-only torch).
:: Run once from the embed\ directory: setup.bat

cd /d "%~dp0"

python -m venv venv
call venv\Scripts\activate.bat

:: CPU-only torch (avoids the 2 GB CUDA download, works on AMD GPU via CPU inference)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Everything else
pip install librosa soundfile noisereduce resemblyzer kokoro

echo.
echo Setup complete. Activate with: venv\Scripts\activate
