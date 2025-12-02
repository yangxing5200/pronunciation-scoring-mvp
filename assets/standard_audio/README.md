# Standard Audio Library

This directory should contain pre-recorded standard pronunciation audio files.

## Structure

```
standard_audio/
├── words/
│   ├── hello.wav
│   ├── world.wav
│   └── ...
├── sentences/
│   ├── sentence_001.wav
│   ├── sentence_002.wav
│   └── ...
└── phonemes/
    ├── ae.wav  # /æ/
    ├── eh.wav  # /ɛ/
    └── ...
```

## Audio Specifications

- **Format**: WAV (uncompressed)
- **Sample Rate**: 16kHz or 22.05kHz
- **Channels**: Mono
- **Bit Depth**: 16-bit PCM

## Generating Standard Audio

### Option 1: Record Native Speakers

Record native speakers pronouncing target texts and save here.

### Option 2: Use TTS (in development environment)

```python
# Example using a TTS system in development
from gtts import gTTS  # or other TTS

text = "Hello world"
tts = gTTS(text, lang='en')
tts.save('standard_audio/sentences/hello_world.wav')
```

### Option 3: Use IndexTTS2 with Standard Voice

Generate standard pronunciations using a high-quality reference voice.

## Usage in Application

The application looks for audio files matching the challenge text:
- Exact filename match
- Normalized text (lowercase, no punctuation)
- Fallback to TTS if not found
