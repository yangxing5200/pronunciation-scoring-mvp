# Example Test Files

This directory would contain example audio files for testing the system.

## Structure

```
examples/
├── audio/
│   ├── good_pronunciation.wav
│   ├── intermediate_pronunciation.wav
│   └── beginner_pronunciation.wav
├── scripts/
│   └── test_pronunciation.py
└── README.md
```

## Sample Test Script

```python
#!/usr/bin/env python3
"""Test pronunciation scoring with sample audio files."""

from core import WhisperTranscriber, PronunciationScorer

# Initialize components
transcriber = WhisperTranscriber(model_size="base")
scorer = PronunciationScorer()

# Test with sample audio
reference_text = "Hello world, this is a test."
audio_path = "examples/audio/good_pronunciation.wav"

# Transcribe
result = transcriber.transcribe(audio_path)

# Score
score_result = scorer.score_pronunciation(
    user_audio_path=audio_path,
    reference_text=reference_text,
    transcribed_text=result["text"],
    word_timestamps=result["words"]
)

print(f"Score: {score_result['total_score']}/100")
print(f"Accuracy: {score_result['accuracy']}")
print(f"Fluency: {score_result['fluency']}")
print(f"Prosody: {score_result['prosody']}")
```

## Creating Test Audio

### Method 1: Record yourself
Use the Streamlit app to record test samples.

### Method 2: Generate with TTS
```python
from gtts import gTTS

text = "Hello world, this is a test."
tts = gTTS(text, lang='en')
tts.save('test_audio.wav')
```

### Method 3: Use existing audio
Convert any audio file to the correct format:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Running Tests

```bash
# Test individual modules
python scripts/test_system.py

# Test with real audio (requires dependencies)
python examples/scripts/test_pronunciation.py
```
