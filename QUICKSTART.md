# Quick Start Guide

## 🚀 5-Minute Setup (Development Environment)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd pronunciation-scoring-mvp
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Download Models
```bash
# Download Whisper model (base recommended for balance)
python scripts/download_models.py --whisper-model base

# This will download ~150MB
```

### 4. Run Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📱 Using the Application

### Step 1: Choose a Challenge
- Select from pre-defined challenges (Hello World, Weather Talk, etc.)
- See the target text and phonetic transcription

### Step 2: Record Your Pronunciation
- **Option A**: Click the microphone button to record (if streamlit-audiorec is installed)
- **Option B**: Upload a pre-recorded audio file (.wav or .mp3)

### Step 3: Analyze
- Click "🔍 Analyze Pronunciation"
- Wait for AI processing (~5-10 seconds)

### Step 4: Review Feedback
- **Overall Score** - Your total pronunciation score (0-100)
- **Breakdown** - Accuracy, Fluency, Prosody scores
- **Word-by-Word** - Color-coded feedback:
  - 🟢 Green = Excellent (≥90)
  - 🟡 Yellow = Good (75-89)
  - 🔴 Red = Needs improvement (<75)
- **Coaching Tips** - Specific advice on what to improve

### Step 5: Voice Cloning (Optional)
- After recording, click "🎨 Generate My Voice"
- Hear the correct pronunciation in YOUR voice
- Compare with your original recording

---

## 🔧 Troubleshooting

### "Models not loaded"
```bash
python scripts/download_models.py --whisper-model base
```

### "streamlit-audiorec not available"
This is optional. You can still upload audio files.
```bash
pip install streamlit-audiorec
```

### "Voice cloning not available"
IndexTTS2 is optional. The app works without it.
To enable:
```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts
pip install -e .
# Download models and place in models/indextts2/
```

### Slow processing
- Use smaller Whisper model: `--whisper-model tiny`
- Check CPU usage (Whisper is computationally intensive)
- Consider using GPU: Edit config.yaml, set `device: "cuda"`

### Low scores
- Ensure good audio quality (16kHz minimum)
- Record in quiet environment
- Speak clearly at natural pace
- Match the reference text exactly

---

## 🎯 Tips for Best Results

### Recording Quality
- Use headset/good microphone
- Minimize background noise
- Speak 6-12 inches from mic
- Avoid plosives (p, b, t, d sounds hitting mic)

### Speaking Technique
- Match reference text exactly
- Speak at natural pace (not too fast/slow)
- Use natural rhythm and intonation
- Don't pause mid-word

### Interpretation
- **Accuracy** - How correct your phonemes are
- **Fluency** - How smooth your speech rhythm is
- **Prosody** - How well your pitch/intonation matches

---

## 📊 Understanding Scores

### Overall Score Ranges
- **90-100**: Excellent - Near-native pronunciation
- **75-89**: Good - Understandable with minor issues
- **60-74**: Fair - Noticeable accent, needs practice
- **Below 60**: Needs improvement - Focus on basics

### Component Scores

**Accuracy (50% weight)**
- Measures phoneme correctness
- Focus: Individual sound production

**Fluency (25% weight)**
- Measures speech rhythm and pauses
- Focus: Natural flow

**Prosody (25% weight)**
- Measures pitch and intonation patterns
- Focus: Melody of speech

---

## 🎓 Practice Recommendations

### Beginners (Scores <60)
1. Focus on individual word pronunciation
2. Use easier challenges first
3. Listen to standard audio repeatedly
4. Practice one sentence at a time

### Intermediate (Scores 60-79)
1. Work on fluency and rhythm
2. Record multiple attempts
3. Compare your voice clone with original
4. Focus on problem words

### Advanced (Scores 80+)
1. Perfect prosody and intonation
2. Try longer, complex sentences
3. Work on subtle phoneme distinctions
4. Aim for consistency across attempts

---

## 📁 File Management

### Where are recordings saved?
- Temporary files: `temp_audio/`
- Auto-deleted on app restart (configurable)

### Can I keep my recordings?
Yes! Before closing the app:
1. Copy files from `temp_audio/`
2. Or modify `config.yaml`: `cleanup_on_exit: false`

### Standard audio library
- Location: `assets/standard_audio/`
- Add your own reference recordings here
- Format: 16kHz WAV files

---

## 🔄 Offline Deployment

For production deployment without internet:

### 1. Prepare (Development Machine)
```bash
# Install everything
pip install -r requirements.txt
python scripts/download_models.py

# Verify it works
streamlit run app.py

# Package for offline
tar -czf pronunciation-scoring-offline.tar.gz \
    app.py core/ models/ assets/ scripts/ \
    requirements.txt README.md config.yaml
```

### 2. Deploy (Offline Machine)
```bash
# Extract
tar -xzf pronunciation-scoring-offline.tar.gz

# Install dependencies (if needed)
pip install -r requirements.txt --no-index --find-links ./wheels/

# Run
streamlit run app.py
```

---

## ❓ FAQ

**Q: Do I need GPU?**
A: No, CPU works fine. GPU makes it faster.

**Q: What audio formats are supported?**
A: .wav, .mp3, .flac

**Q: Can I add more languages?**
A: Yes! Edit `config.yaml` and update language settings.

**Q: Is my data sent anywhere?**
A: No! Everything runs locally on your machine.

**Q: Can I customize scoring weights?**
A: Yes! Edit `config.yaml` scoring section.

---

## 🆘 Getting Help

- Run system test: `python scripts/test_system.py`
- Check logs in terminal where you ran `streamlit run app.py`
- Review README.md for detailed documentation
- Check model files in `models/` directories

---

## 🎉 Next Steps

1. ✅ Complete quick start above
2. 📖 Read full README.md for advanced features
3. 🎯 Try different challenges
4. 📈 Track your improvement over time
5. 🔧 Customize config.yaml to your needs
