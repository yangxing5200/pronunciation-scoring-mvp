# Pronunciation Scoring MVP

A complete offline-friendly AI-powered pronunciation assessment system supporting real-time recording, speech analysis, and personalized feedback.

## Features

- 🎙️ **Real-time Audio Recording** - Record directly in the browser using streamlit-audiorec
- 🔊 **Standard Pronunciation Playback** - Built-in standard pronunciation library
- ✨ **Voice Cloning** - Hear standard pronunciation in your own voice (using IndexTTS2)
- 📊 **Comprehensive Scoring** - Three-dimensional assessment:
  - **Accuracy** - Phoneme-level pronunciation correctness
  - **Fluency** - Speech rhythm and pause analysis
  - **Prosody** - Pitch and intonation patterns
- 🎯 **Detailed Feedback** - Word-by-word color-coded feedback with specific improvement tips
- 🔌 **Fully Offline** - All AI models run locally, no internet required in production

## Technical Architecture

### Core Components

```
pronunciation-scoring-mvp/
├── app.py                      # Streamlit main application
├── core/                       # Core processing modules
│   ├── transcriber.py         # Whisper-based speech recognition
│   ├── aligner.py             # Phoneme alignment (DTW)
│   ├── scorer.py              # 3D scoring system
│   ├── voice_cloner.py        # IndexTTS2 voice cloning
│   └── text_comparator.py     # Text comparison (Levenshtein)
├── models/                     # Pre-downloaded AI models
│   ├── whisper/               # Whisper models (base/small)
│   ├── indextts2/             # IndexTTS2 voice cloning model
│   └── alignment/             # Alignment models (if needed)
├── assets/
│   └── standard_audio/        # Pre-generated standard pronunciations
├── scripts/
│   └── download_models.py     # Model download script
└── requirements.txt            # Python dependencies
```

### AI Models Used

1. **Whisper** (OpenAI) - Speech recognition with word-level timestamps
2. **IndexTTS2** - Voice cloning for timbre transfer
3. **librosa** - Audio feature extraction (pitch, energy, MFCC)
4. **DTW** - Dynamic Time Warping for alignment

## Installation

> **📖 For detailed installation instructions, troubleshooting, and system requirements, see [INSTALL.md](INSTALL.md)**

### Prerequisites

- **Python 3.10 or higher** (tested with Python 3.10, 3.11, and 3.12)
- **pip** (latest version recommended)
- **Git** (for cloning repositories)
- **CUDA 11.8** (optional, only if using GPU acceleration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd pronunciation-scoring-mvp
```

### Step 2: Create Virtual Environment (Recommended)

Creating a virtual environment helps avoid dependency conflicts:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

Always use the latest pip to avoid installation issues:

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

Choose the installation method based on your hardware:

#### Option A: CPU-Only Installation (No GPU Required)

This is suitable for most users and doesn't require NVIDIA GPU:

```bash
pip install -r requirements-cpu.txt
```

**Note:** CPU inference is slower but works on all machines.

#### Option B: GPU Installation (Recommended for NVIDIA GPU users)

If you have an NVIDIA GPU with CUDA 11.8 support:

```bash
pip install -r requirements-cuda.txt
```

**Benefits:**
- 5-10x faster inference
- Better for real-time processing
- Supports larger Whisper models

**Requirements:**
- NVIDIA GPU (GTX 1060 or better recommended)
- CUDA 11.8 installed
- At least 4GB VRAM

#### Option C: Alternative - Base Installation

For simple setup, you can also use:

```bash
pip install -r requirements.txt
```

This installs CPU versions by default.

### Step 5: Install IndexTTS2 (Optional, for Voice Cloning)

IndexTTS2 enables the voice cloning feature. If you skip this step, the system will work in fallback mode using pre-generated standard audio.

```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts
pip install -e .
cd ..
```

Download IndexTTS2 pre-trained models and place them in `models/indextts2/`. See the [IndexTTS2 repository](https://github.com/index-tts/index-tts) for model download instructions.

### Step 6: Download AI Models

Download the Whisper model for speech recognition:

```bash
python scripts/download_models.py --whisper-model base
```

**Available Whisper models:**
- `tiny` - Fastest, ~39M params (~1 GB RAM)
- `base` - **Recommended**, ~74M params (~1.5 GB RAM)
- `small` - Better accuracy, ~244M params (~2.5 GB RAM)
- `medium` - High accuracy, ~769M params (~5 GB RAM, GPU recommended)
- `large` - Best accuracy, ~1550M params (~10 GB RAM, GPU required)

### Step 7: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Troubleshooting Installation

#### Issue: Dependency conflicts during installation

**Solution:**
1. Make sure you're using Python 3.10 or higher: `python --version`
2. Upgrade pip: `python -m pip install --upgrade pip`
3. Use a fresh virtual environment
4. Install from the correct requirements file (cpu vs cuda)

#### Issue: "cannot import name 'builder' from 'google.protobuf.internal'"

**Solution:** This means protobuf version is too old. Update it:
```bash
pip install --upgrade "protobuf>=4.25.0,<5.0.0"
```

#### Issue: Torch version conflicts

**Solution:** Uninstall all torch packages and reinstall from requirements:
```bash
pip uninstall torch torchvision torchaudio -y
pip install -r requirements-cpu.txt  # or requirements-cuda.txt
```

#### Issue: numpy version conflicts with dtw-python

**Solution:** The requirements files specify compatible versions. If issues persist:
```bash
pip install numpy==1.26.4 dtw-python==1.5.1
```

#### Issue: GPU not detected (CUDA installation)

**Solution:**
1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
3. If false, reinstall with: `pip install -r requirements-cuda.txt --force-reinstall`

#### Issue: Packages requiring system libraries (faster-whisper, phonemizer, etc.)

Some packages may require system libraries to be installed first:

**For faster-whisper (requires ffmpeg):**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

**For phonemizer (requires espeak-ng):**
```bash
# Ubuntu/Debian
sudo apt install espeak-ng

# macOS
brew install espeak-ng
```

**Note:** See [INSTALL.md](INSTALL.md) for detailed system dependency information.

### Offline Environment (Production Deployment)

For deploying to a machine without internet access:

1. **Prepare in development environment** (with internet access)
   - Complete all installation steps above
   - Ensure all models are downloaded to `models/` directory
   - Verify the application works: `streamlit run app.py`
   - Download wheel files for offline installation:
     ```bash
     mkdir wheels
     pip download -r requirements-cpu.txt -d wheels/
     # Or for GPU: pip download -r requirements-cuda.txt -d wheels/
     ```

2. **Package for offline deployment**
   ```bash
   # Create a deployment package
   tar -czf pronunciation-scoring-offline.tar.gz \
       app.py \
       core/ \
       models/ \
       assets/ \
       scripts/ \
       requirements.txt \
       requirements-cpu.txt \
       requirements-cuda.txt \
       wheels/ \
       README.md
   ```

3. **Deploy to offline environment**
   ```bash
   # On offline machine
   tar -xzf pronunciation-scoring-offline.tar.gz
   cd pronunciation-scoring-mvp
   
   # Create virtual environment
   python -m venv venv
   # Activate (Windows: venv\Scripts\activate, Linux/Mac: source venv/bin/activate)
   
   # Install packages from local wheels
   pip install --upgrade pip
   pip install --no-index --find-links ./wheels/ -r requirements-cpu.txt
   # Or for GPU: pip install --no-index --find-links ./wheels/ -r requirements-cuda.txt
   
   # Run application
   streamlit run app.py
   ```

## Usage

### Basic Workflow

1. **Select a practice sentence** - Choose from built-in challenges or custom text
2. **Listen to standard pronunciation** - Hear native speaker pronunciation
3. **Record your pronunciation** - Click record button (5-second max)
4. **Get instant feedback** - Receive detailed scoring and suggestions
5. **Listen to your voice saying it correctly** - AI generates standard pronunciation in your voice

### Scoring System

The system provides scores in three dimensions:

#### 1. Pronunciation Accuracy (0-100)
- Compares your phonemes with standard pronunciation
- Detects substitutions, omissions, and distortions
- Example: /i/ → /ɪ/ gives partial credit, /r/ → /l/ gives lower score

#### 2. Fluency (0-100)
- Analyzes speech rhythm and pauses
- Detects choppy speech or unnatural hesitations
- Measures speaking rate consistency

#### 3. Prosody (0-100)
- Compares pitch patterns with native speakers
- Analyzes intonation and stress patterns
- Measures energy distribution across syllables

### Feedback Interface

- **Overall Score** - Large display at top (0-100)
- **Word-level Visualization** - Color-coded words:
  - 🟢 Green: Excellent (≥90)
  - 🟡 Yellow: Good (75-89)
  - 🔴 Red: Needs improvement (<75)
- **Specific Issues** - Top 3 problems with actionable advice
- **Detailed Metrics** - Breakdown by accuracy, fluency, prosody

## Configuration

### Model Selection

Edit `core/transcriber.py` to change Whisper model size:

```python
transcriber = WhisperTranscriber(
    model_size="base",  # Options: tiny, base, small, medium, large
    model_dir="models/whisper"
)
```

**Model Size Trade-offs:**
- `tiny` - Fastest, least accurate (~39M params)
- `base` - **Recommended** for balance (~74M params)
- `small` - Better accuracy, slower (~244M params)
- `medium` - High accuracy, requires more RAM (~769M params)
- `large` - Best accuracy, requires GPU (~1550M params)

### Audio Settings

In `core/aligner.py`, adjust sample rate:

```python
aligner = PhonemeAligner(sample_rate=16000)  # 16kHz is standard
```

## File Formats

### Supported Audio Input
- `.wav` - Recommended (16kHz, mono)
- `.mp3` - Supported
- `.flac` - Supported

### Output Files
- Recordings saved to temp directory (auto-cleaned)
- Voice clones saved to `assets/cloned_audio/`

## Troubleshooting

### "Whisper model not found"
```bash
python scripts/download_models.py --whisper-model base
```

### "IndexTTS2 not available"
Voice cloning will work in fallback mode. To enable:
1. Install IndexTTS2 from https://github.com/index-tts/index-tts
2. Download pre-trained models
3. Place in `models/indextts2/`

### Low accuracy scores
- Check audio quality (16kHz minimum)
- Ensure quiet recording environment
- Speak clearly and at normal pace
- Verify reference text matches recording

### Slow processing
- Use smaller Whisper model (`tiny` or `base`)
- Disable voice cloning for faster feedback
- Use GPU if available (set `device="cuda"`)

## Development

### Project Structure

```python
# Core modules
from core import (
    WhisperTranscriber,    # Speech recognition
    PhonemeAligner,        # DTW alignment
    PronunciationScorer,   # Scoring system
    TextComparator,        # Text comparison
    VoiceCloner           # Voice cloning
)
```

### Adding Custom Scoring Logic

Edit `core/scorer.py`:

```python
def _calculate_accuracy(self, text_comparison, word_scores):
    # Add custom logic here
    pass
```

### Adding New Languages

1. Update Whisper language parameter in `core/transcriber.py`
2. Add language-specific phoneme mappings in `core/scorer.py`
3. Add standard audio files for the language

## Performance Optimization

### For Production
- Use `base` Whisper model (good balance)
- Pre-generate standard pronunciations
- Enable FP16 for GPU inference
- Disable voice cloning if not needed

### For Development
- Use `tiny` Whisper model for faster iteration
- Use smaller test audio files
- Enable verbose logging

## Security Notes

- All models run locally - no data sent to external servers
- Audio files stored temporarily and can be auto-deleted
- No API keys or credentials required
- Safe for sensitive/private recordings

## License

[Add your license here]

## Credits

- **Whisper** - OpenAI
- **IndexTTS2** - Index-TTS team
- **librosa** - Audio analysis library
- **streamlit-audiorec** - Audio recording component

## Contributing

[Add contribution guidelines]

## Support

For issues and questions:
- GitHub Issues: [repository URL]
- Documentation: [docs URL]

## Changelog

### v1.0.0 (Current)
- ✅ Offline Whisper integration
- ✅ Real-time audio recording
- ✅ Three-dimensional scoring system
- ✅ Word-level feedback
- ✅ IndexTTS2 voice cloning support
- ✅ Complete offline capability
