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

### Development Environment (with Internet)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pronunciation-scoring-mvp
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download AI models**
   ```bash
   python scripts/download_models.py --whisper-model base
   ```

4. **Install IndexTTS2 (optional, for voice cloning)**
   ```bash
   git clone https://github.com/index-tts/index-tts.git
   cd index-tts
   pip install -e .
   cd ..
   ```
   
   Download IndexTTS2 pre-trained models and place in `models/indextts2/`
   
   **Note**: Voice cloning will work in fallback mode without IndexTTS2

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Offline Environment (Production Deployment)

1. **Prepare in development environment** (with internet)
   - Complete all installation steps above
   - Ensure all models are downloaded to `models/` directory
   - Verify the application works: `streamlit run app.py`

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
       README.md
   ```

3. **Deploy to offline environment**
   ```bash
   # On offline machine
   tar -xzf pronunciation-scoring-offline.tar.gz
   cd pronunciation-scoring-mvp
   
   # Install Python packages (if not pre-installed)
   pip install -r requirements.txt --no-index --find-links ./wheels/
   
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
