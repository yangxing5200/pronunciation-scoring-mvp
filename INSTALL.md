# Installation Guide

This document provides detailed installation instructions for the Pronunciation Scoring MVP system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation Steps](#detailed-installation-steps)
4. [Package-Specific Notes](#package-specific-notes)
5. [Troubleshooting](#troubleshooting)
6. [Verification](#verification)

## System Requirements

### Python Version
- **Python 3.10, 3.11, or 3.12** (recommended: 3.11)
- Python 3.9 and earlier are **not supported**

### Operating Systems
- **Linux**: Tested on Ubuntu 20.04+, Debian 11+
- **macOS**: Tested on macOS 11+ (Big Sur and later)
- **Windows**: Tested on Windows 10/11

### Hardware Requirements

**Minimum (CPU-only):**
- 4 GB RAM
- 2 GB free disk space
- Any modern CPU (x86_64)

**Recommended (with GPU):**
- 8 GB RAM
- 4 GB VRAM (GPU)
- 5 GB free disk space
- NVIDIA GPU with CUDA 11.8+ support (GTX 1060 or better)

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd pronunciation-scoring-mvp

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements-cpu.txt   # For CPU-only
# OR
pip install -r requirements-cuda.txt  # For GPU with CUDA

# 6. Download models
python scripts/download_models.py --whisper-model base

# 7. Run the application
streamlit run app.py
```

## Detailed Installation Steps

### Step 1: Verify Python Installation

```bash
python --version
```

Expected output: `Python 3.10.x`, `Python 3.11.x`, or `Python 3.12.x`

If you don't have Python 3.10+, download it from:
- **Official**: https://www.python.org/downloads/
- **Ubuntu/Debian**: `sudo apt install python3.11 python3.11-venv`
- **macOS (Homebrew)**: `brew install python@3.11`
- **Windows**: Download from python.org or use Microsoft Store

### Step 2: Create Virtual Environment

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system packages
- Makes the project portable

```bash
# Create virtual environment
python -m venv venv

# Or with a specific Python version
python3.11 -m venv venv
```

### Step 3: Activate Virtual Environment

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Note:** If you get an execution policy error on Windows PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Upgrade pip

Always use the latest pip to avoid installation issues:

```bash
python -m pip install --upgrade pip
```

### Step 5: Install Dependencies

#### Option A: CPU-Only Installation

Best for users without NVIDIA GPUs or for development/testing:

```bash
pip install -r requirements-cpu.txt
```

**Installation time:** ~5-10 minutes depending on your internet connection

#### Option B: GPU Installation (CUDA)

For users with NVIDIA GPUs:

```bash
pip install -r requirements-cuda.txt
```

**Prerequisites:**
- NVIDIA GPU with compute capability 3.5+ (check: https://developer.nvidia.com/cuda-gpus)
- CUDA Toolkit 11.8 or later (download: https://developer.nvidia.com/cuda-downloads)
- cuDNN 8.0+ (included with PyTorch)

**Verify GPU installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`

### Step 6: Install Optional Components

#### IndexTTS2 (Voice Cloning)

IndexTTS2 is optional but enables the voice cloning feature:

```bash
# Clone the repository
git clone https://github.com/index-tts/index-tts.git
cd index-tts

# Install in development mode
pip install -e .

# Return to project directory
cd ..
```

**Note:** Download pre-trained models from the IndexTTS2 repository and place them in `models/indextts2/`

### Step 7: Download AI Models

```bash
python scripts/download_models.py --whisper-model base
```

**Model options:**
- `tiny`: 39M params, ~1 GB RAM, fastest
- `base`: 74M params, ~1.5 GB RAM, **recommended**
- `small`: 244M params, ~2.5 GB RAM, better accuracy
- `medium`: 769M params, ~5 GB RAM, high accuracy (GPU recommended)
- `large`: 1550M params, ~10 GB RAM, best accuracy (GPU required)

### Step 8: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at http://localhost:8501

## Package-Specific Notes

### Packages Requiring System Libraries

Some packages may need system libraries to be installed:

#### faster-whisper

May require ffmpeg for audio processing:

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

#### praat-parselmouth

May require additional build tools on some systems:

**Ubuntu/Debian:**
```bash
sudo apt install build-essential
```

**macOS:**
```bash
xcode-select --install
```

#### phonemizer

Requires espeak-ng:

**Ubuntu/Debian:**
```bash
sudo apt install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows:**
Download from https://github.com/espeak-ng/espeak-ng/releases

## Troubleshooting

### Issue: "python: command not found"

**Solution:** Use `python3` instead of `python`:
```bash
python3 --version
python3 -m venv venv
```

### Issue: "pip: command not found"

**Solution:** Install pip or use `python -m pip`:
```bash
# Ubuntu/Debian
sudo apt install python3-pip

# Or use module syntax
python -m pip install --upgrade pip
```

### Issue: Dependency conflicts during installation

**Solution:**
1. Use a fresh virtual environment
2. Make sure pip is updated
3. Install from the correct requirements file

```bash
# Remove old venv
rm -rf venv

# Create fresh venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Update pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements-cpu.txt
```

### Issue: "cannot import name 'builder' from 'google.protobuf.internal'"

**Cause:** Old protobuf version (3.19.6 or earlier)

**Solution:**
```bash
pip install --upgrade "protobuf>=4.25.0,<5.0.0"
```

### Issue: numpy version conflicts

**Cause:** Incompatible dtw-python version

**Solution:**
```bash
pip install numpy==1.26.4 dtw-python==1.5.1
```

### Issue: torch/torchvision version mismatch

**Solution:** Uninstall all torch packages and reinstall:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
```

### Issue: GPU not detected (CUDA installation)

**Solution:**
1. Verify NVIDIA driver: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Verify PyTorch GPU support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
   ```
4. If false, reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install -r requirements-cuda.txt --force-reinstall
   ```

### Issue: Slow model download

**Solution:** The models are downloaded from the internet. If downloads are slow:
1. Use a VPN or proxy if certain domains are blocked
2. Download models manually and place in `models/whisper/`
3. Use `--whisper-model tiny` for a smaller model first

## Verification

After installation, verify everything works:

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Check all packages installed
pip check

# Verify key packages
python -c "
import streamlit
import numpy as np
import torch
import librosa
from google.protobuf import descriptor

print('✓ Streamlit:', streamlit.__version__)
print('✓ NumPy:', np.__version__)
print('✓ PyTorch:', torch.__version__)
print('✓ Librosa:', librosa.__version__)
print('✓ Protobuf: OK')
print('✓ CUDA available:', torch.cuda.is_available())
print('\nAll dependencies installed successfully!')
"

# Run the application
streamlit run app.py
```

Expected output:
```
✓ Streamlit: 1.39.1
✓ NumPy: 1.26.4
✓ PyTorch: 2.2.2
✓ Librosa: 0.10.1
✓ Protobuf: OK
✓ CUDA available: True/False

All dependencies installed successfully!
```

## Additional Resources

- **Project README**: See `README.md` for feature overview
- **Model Download**: `python scripts/download_models.py --help`
- **Streamlit Docs**: https://docs.streamlit.io
- **PyTorch Docs**: https://pytorch.org/docs/stable/index.html
- **Whisper Docs**: https://github.com/openai/whisper

## Getting Help

If you encounter issues not covered in this guide:

1. Check the GitHub Issues page
2. Review the troubleshooting section
3. Ensure you're using Python 3.10+
4. Try with a fresh virtual environment
5. Check that all system dependencies are installed

## License

See LICENSE file in the repository root.
