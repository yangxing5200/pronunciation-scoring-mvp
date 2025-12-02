"""
Whisper-based speech transcription module for offline use.
"""

import os
import warnings
from typing import Dict, List, Optional, Union
import torch
import whisper
from pathlib import Path


class WhisperTranscriber:
    """
    Offline Whisper transcriber with word-level timestamps.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        model_dir: str = "models/whisper",
        device: Optional[str] = None,
        language: str = "en"
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
            model_dir: Directory containing downloaded Whisper models
            device: Device to use (cuda/cpu), auto-detect if None
            language: Target language code
        """
        self.model_size = model_size
        self.model_dir = Path(model_dir)
        self.language = language
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model from local directory."""
        try:
            # Ensure model directory exists
            self.model_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to load from local directory first
            model_path = self.model_dir / f"{self.model_size}.pt"
            
            if model_path.exists():
                print(f"Loading Whisper model from {model_path}")
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=str(self.model_dir)
                )
            else:
                # In offline environment, this should fail
                # In development, it will download
                warnings.warn(
                    f"Model {self.model_size} not found at {model_path}. "
                    f"Attempting to download (requires internet)."
                )
                self.model = whisper.load_model(
                    self.model_size,
                    device=self.device,
                    download_root=str(self.model_dir)
                )
            
            print(f"Whisper model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Whisper model: {e}\n"
                f"Please run 'python scripts/download_models.py' first."
            )
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Override language (uses self.language if None)
        
        Returns:
            Dictionary containing:
                - text: Full transcription
                - segments: List of segments with timestamps
                - words: List of word-level timestamps
                - language: Detected/specified language
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        target_lang = language or self.language
        
        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            str(audio_path),
            language=target_lang,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract word-level information
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info.get("word", "").strip(),
                    "start": word_info.get("start", 0.0),
                    "end": word_info.get("end", 0.0),
                    "probability": word_info.get("probability", 1.0)
                })
        
        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "words": words,
            "language": result.get("language", target_lang)
        }
    
    def get_word_timestamps(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Get word-level timestamps from audio.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            List of dictionaries with word, start, end, probability
        """
        result = self.transcribe(audio_path)
        return result["words"]
    
    def compare_with_reference(
        self,
        audio_path: Union[str, Path],
        reference_text: str
    ) -> Dict:
        """
        Compare transcription with reference text.
        
        Args:
            audio_path: Path to audio file
            reference_text: Expected text
        
        Returns:
            Dictionary with transcription and comparison info
        """
        result = self.transcribe(audio_path)
        transcribed = result["text"].lower().strip()
        reference = reference_text.lower().strip()
        
        return {
            "transcribed": transcribed,
            "reference": reference,
            "match": transcribed == reference,
            "words": result["words"],
            "segments": result["segments"]
        }
