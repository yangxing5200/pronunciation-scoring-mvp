# Deployment Checklist

Use this checklist to ensure proper deployment of the Pronunciation Scoring MVP.

## 📋 Pre-Deployment (Development Environment)

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed: `pip install -r requirements.txt`

### Model Downloads
- [ ] Whisper model downloaded (base recommended)
- [ ] Models stored in `models/whisper/`
- [ ] IndexTTS2 installed (optional for voice cloning)
- [ ] IndexTTS2 models in `models/indextts2/` (optional)

### Testing
- [ ] Run system test: `python scripts/test_system.py`
- [ ] Application starts: `streamlit run app.py`
- [ ] Can upload audio file
- [ ] Transcription works
- [ ] Scoring works
- [ ] Results display correctly

### Optional Features
- [ ] streamlit-audiorec installed and working
- [ ] Voice cloning functional (if IndexTTS2 installed)
- [ ] Standard audio library populated

## 📦 Packaging for Offline Deployment

### 1. Verify Complete Installation
```bash
# Ensure all models are downloaded
ls -lh models/whisper/
# Should see .pt files

# Verify dependencies
pip freeze > installed_packages.txt
```

### 2. Create Offline Package
```bash
# Create deployment archive
tar -czf pronunciation-scoring-offline.tar.gz \
    app.py \
    core/ \
    models/ \
    assets/ \
    scripts/ \
    examples/ \
    requirements.txt \
    config.yaml \
    README.md \
    QUICKSTART.md \
    DEPLOYMENT.md

# Verify archive
tar -tzf pronunciation-scoring-offline.tar.gz | head -20
```

### 3. Optional: Include Python Packages
```bash
# Download all packages as wheels
pip download -r requirements.txt -d wheels/

# Include in archive
tar -czf pronunciation-scoring-full-offline.tar.gz \
    app.py core/ models/ assets/ scripts/ examples/ \
    requirements.txt config.yaml README.md QUICKSTART.md \
    wheels/
```

## 🚀 Offline Deployment

### On Target Machine

#### 1. Extract Package
```bash
tar -xzf pronunciation-scoring-offline.tar.gz
cd pronunciation-scoring-mvp
```

#### 2. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

**Option A: With internet (if available)**
```bash
pip install -r requirements.txt
```

**Option B: From local wheels (no internet)**
```bash
pip install --no-index --find-links ./wheels/ -r requirements.txt
```

**Option C: Pre-installed Python packages**
If Python packages are already installed system-wide, skip this step.

#### 4. Verify Installation
```bash
# Run system test
python scripts/test_system.py

# Should show all tests passing
```

#### 5. Start Application
```bash
streamlit run app.py
```

## ✅ Post-Deployment Verification

### Functional Tests
- [ ] Application starts without errors
- [ ] UI loads in browser
- [ ] Can select challenges
- [ ] Can upload audio files
- [ ] Audio processing works
- [ ] Scoring results display
- [ ] Word-by-word feedback shows
- [ ] Color coding works

### Performance Tests
- [ ] Transcription completes in <30 seconds
- [ ] Scoring completes in <10 seconds
- [ ] No memory errors
- [ ] Application responsive

### Optional Features
- [ ] Audio recording works (if streamlit-audiorec installed)
- [ ] Voice cloning works (if IndexTTS2 installed)
- [ ] Standard audio playback works

## 🔧 Configuration

### Customize for Target Environment

#### Edit config.yaml:
```yaml
# Adjust based on hardware
whisper:
  model_size: "base"  # Use "tiny" for slower machines
  device: "cpu"       # Change to "cuda" if GPU available

# Adjust scoring weights if needed
scoring:
  accuracy_weight: 0.5
  fluency_weight: 0.25
  prosody_weight: 0.25
```

## 📊 Production Considerations

### Performance Optimization
- [ ] Use appropriate Whisper model size for hardware
- [ ] Enable GPU if available
- [ ] Configure streamlit for production
- [ ] Set up logging
- [ ] Configure auto-cleanup of temp files

### Security
- [ ] Review file permissions
- [ ] Configure network access (if needed)
- [ ] Set up user authentication (if multi-user)
- [ ] Review data retention policies

### Monitoring
- [ ] Set up application logging
- [ ] Monitor disk space (temp files)
- [ ] Monitor memory usage
- [ ] Track processing times

## 🆘 Troubleshooting

### Common Issues

**"Models not found"**
- Verify models/ directory contains downloaded models
- Check file permissions
- Run: `ls -la models/whisper/`

**"Import Error"**
- Verify all dependencies installed
- Check Python version (3.8+)
- Run: `pip list | grep -E "torch|whisper|streamlit"`

**Slow Performance**
- Use smaller Whisper model (tiny/base)
- Check available RAM
- Close other applications
- Consider GPU acceleration

**UI Not Loading**
- Check firewall settings
- Try different port: `streamlit run app.py --server.port 8502`
- Check browser console for errors

## 📝 Documentation Checklist

Ensure users have access to:
- [ ] README.md - Full documentation
- [ ] QUICKSTART.md - Quick start guide
- [ ] This DEPLOYMENT.md - Deployment instructions
- [ ] config.yaml - Configuration reference
- [ ] examples/README.md - Example usage

## 🎯 Success Criteria

Deployment is successful when:
- ✅ Application starts without errors
- ✅ Users can upload/record audio
- ✅ Pronunciation analysis completes
- ✅ Scores are displayed with feedback
- ✅ System is stable over extended use
- ✅ Performance is acceptable for target hardware

## 📞 Support Resources

- System test: `python scripts/test_system.py`
- Check configuration: `cat config.yaml`
- View logs: Check terminal where streamlit runs
- Model verification: `ls -lh models/whisper/ models/indextts2/`

---

## Version History

- v1.0.0 - Initial offline-capable release
  - Whisper integration
  - Three-dimensional scoring
  - Optional voice cloning
  - Complete offline support
