#!/usr/bin/env python3
"""
Model download script for offline deployment.

This script downloads all necessary models for the pronunciation scoring system.
Run this in a development environment with internet access.

Usage:
    python scripts/download_models.py [--whisper-model base] [--device cpu]
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_whisper_model(model_size: str = "base", model_dir: str = "models/whisper"):
    """
    Download Whisper model.
    
    Args:
        model_size: Whisper model size (tiny, base, small, medium, large)
        model_dir: Directory to save model
    """
    print(f"\n{'='*60}")
    print(f"Downloading Whisper model: {model_size}")
    print(f"{'='*60}")
    
    try:
        import whisper
        
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading to: {model_path.absolute()}")
        model = whisper.load_model(model_size, download_root=str(model_path))
        print(f"✓ Whisper {model_size} model downloaded successfully")
        
        # Verify model file exists
        model_file = model_path / f"{model_size}.pt"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"  Model file: {model_file}")
            print(f"  Size: {size_mb:.2f} MB")
        
        return True
    except Exception as e:
        print(f"✗ Failed to download Whisper model: {e}")
        return False


def download_indextts2_instructions():
    """
    Print instructions for downloading IndexTTS2.
    
    IndexTTS2 is not available via pip and requires manual setup.
    """
    print(f"\n{'='*60}")
    print("IndexTTS2 Model Setup Instructions")
    print(f"{'='*60}")
    
    print("""
IndexTTS2 requires manual installation and model download:

1. Install IndexTTS2:
   git clone https://github.com/index-tts/index-tts.git
   cd index-tts
   pip install -e .

2. Download pre-trained models:
   Follow instructions at: https://github.com/index-tts/index-tts
   
3. Place model files in: models/indextts2/
   Expected structure:
   models/indextts2/
   ├── config.yaml
   ├── checkpoints/
   └── [other model files]

Note: Voice cloning will work in fallback mode if IndexTTS2 is not available.
The system will use pre-generated standard audio files instead.
""")


def create_model_structure():
    """Create necessary model directory structure."""
    print(f"\n{'='*60}")
    print("Creating model directory structure")
    print(f"{'='*60}")
    
    directories = [
        "models/whisper",
        "models/indextts2",
        "models/alignment",
        "assets/standard_audio",
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {path.absolute()}")
    
    # Create placeholder config for IndexTTS2
    config_path = Path("models/indextts2/config.yaml")
    if not config_path.exists():
        # Create a properly formatted YAML config
        config_content = """\
# IndexTTS2 Configuration Placeholder
# Replace this with actual config.yaml from IndexTTS2 model download
model_type: indextts2
sample_rate: 22050
"""
        config_path.write_text(config_content)
        print(f"✓ Created placeholder config: {config_path}")


def create_readme_in_models():
    """Create README files in model directories."""
    print(f"\n{'='*60}")
    print("Creating README files in model directories")
    print(f"{'='*60}")
    
    whisper_readme = Path("models/whisper/README.md")
    whisper_readme.write_text("""# Whisper Models

This directory contains downloaded Whisper models for offline speech recognition.

## Downloaded Models

Run `python scripts/download_models.py` to download models.

## Model Sizes

- tiny: ~39M parameters, fastest
- base: ~74M parameters, balanced (recommended)
- small: ~244M parameters, better accuracy
- medium: ~769M parameters, high accuracy
- large: ~1550M parameters, best accuracy (requires GPU)

## Files

After download, you should see:
- base.pt (or other model size)
""")
    
    indextts_readme = Path("models/indextts2/README.md")
    indextts_readme.write_text("""# IndexTTS2 Models

This directory should contain IndexTTS2 model files for voice cloning.

## Setup Instructions

1. Clone IndexTTS2 repository
2. Download pre-trained models
3. Place model files here

See: https://github.com/index-tts/index-tts

## Expected Structure

```
models/indextts2/
├── config.yaml
├── checkpoints/
│   └── [checkpoint files]
└── [other model files]
```
""")
    
    print(f"✓ Created: {whisper_readme}")
    print(f"✓ Created: {indextts_readme}")


def verify_dependencies():
    """Verify that required dependencies are installed."""
    print(f"\n{'='*60}")
    print("Verifying dependencies")
    print(f"{'='*60}")
    
    required_packages = [
        ("whisper", "openai-whisper"),
        ("torch", "torch"),
        ("librosa", "librosa"),
        ("streamlit", "streamlit"),
    ]
    
    missing = []
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not found")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download models for offline pronunciation scoring"
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size to download (default: base)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip dependency verification"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pronunciation Scoring MVP - Model Download")
    print("=" * 60)
    
    # Verify dependencies
    if not args.skip_verification:
        if not verify_dependencies():
            print("\n⚠ Please install missing dependencies first:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    
    # Create directory structure
    create_model_structure()
    
    # Download Whisper model
    success = download_whisper_model(args.whisper_model, "models/whisper")
    
    # IndexTTS2 instructions
    download_indextts2_instructions()
    
    # Create README files
    create_readme_in_models()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if success:
        print("✓ Whisper model downloaded successfully")
    else:
        print("✗ Whisper model download failed")
    
    print("""
Next steps:
1. Install IndexTTS2 manually (see instructions above)
2. Run the application: streamlit run app.py
3. For offline deployment, copy the entire project including models/ directory
""")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
