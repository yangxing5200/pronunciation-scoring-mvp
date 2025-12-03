"""
Task 2: Audio Alignment (Forced Alignment)

Aligns audio with text pinyin to get timestamps for each character/phoneme.
Uses WhisperX for offline, GPU-optimized alignment.
"""

from typing import List, Dict, Optional
from pathlib import Path
import warnings


class ChineseAudioAligner:
    """
    Aligns Chinese audio with text using WhisperX forced alignment.
    
    Provides character-level timestamps for Chinese pronunciation scoring.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize audio aligner.
        
        Args:
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.device = device
        self.whisperx_available = False
        self.model = None
        self.align_model = None
        self.align_metadata = None
        
        # Try to import and initialize WhisperX
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            import whisperx
            self.whisperx = whisperx
            self.whisperx_available = True
        except ImportError:
            warnings.warn(
                "WhisperX not available. Install with: "
                "pip install git+https://github.com/m-bain/whisperx.git"
            )
    
    def load_models(self, model_size: str = "base"):
        """
        Load WhisperX models for transcription and alignment.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        if not self.whisperx_available:
            raise RuntimeError("WhisperX not available")
        
        # Load WhisperX transcription model
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = self.whisperx.load_model(
            model_size,
            self.device,
            compute_type=compute_type
        )
        
        # Load alignment model for Chinese
        self.align_model, self.align_metadata = self.whisperx.load_align_model(
            language_code="zh",
            device=self.device
        )
        
        print(f"WhisperX models loaded on {self.device}")
    
    def align_audio(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Align audio with pinyin sequence to get character-level timestamps.
        
        Args:
            audio_path: Path to audio file
            pinyin_sequence: List from PinyinMapper with 'char' and 'pinyin' keys
        
        Returns:
            List of alignment results with timestamps:
            [
                {"char":"你", "pinyin":"ni3", "start":0.12, "end":0.36},
                {"char":"好", "pinyin":"hao3", "start":0.36, "end":0.58},
                ...
            ]
        """
        if not self.whisperx_available:
            raise RuntimeError("WhisperX not available")
        
        if self.model is None or self.align_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Load audio
        audio = self.whisperx.load_audio(audio_path)
        
        # Transcribe with WhisperX
        result = self.model.transcribe(
            audio,
            batch_size=16,
            language="zh"
        )
        
        # Align for character-level timestamps
        aligned_result = self.whisperx.align(
            result["segments"],
            self.align_model,
            self.align_metadata,
            audio,
            self.device,
            return_char_alignments=True  # Get character-level for Chinese
        )
        
        # Extract character-level timestamps
        char_timestamps = []
        
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                # For Chinese, WhisperX provides character alignments
                if "chars" in word_info:
                    for char_info in word_info["chars"]:
                        char = char_info.get("char", "").strip()
                        if char:  # Only add non-empty characters
                            char_timestamps.append({
                                "char": char,
                                "start": char_info.get("start", 0.0),
                                "end": char_info.get("end", 0.0),
                                "score": char_info.get("score", 1.0)
                            })
                else:
                    # Fallback: if no character alignment, use word as character
                    char = word_info.get("word", "").strip()
                    if char:
                        char_timestamps.append({
                            "char": char,
                            "start": word_info.get("start", 0.0),
                            "end": word_info.get("end", 0.0),
                            "score": word_info.get("score", 1.0)
                        })
        
        # Match with expected pinyin sequence
        aligned_result_list = []
        
        # Create mapping of expected characters
        expected_chars = [item["char"] for item in pinyin_sequence]
        pinyin_map = {item["char"]: item["pinyin"] for item in pinyin_sequence}
        
        # Match detected characters with expected sequence
        for i, expected_char in enumerate(expected_chars):
            # Find matching character in timestamps
            matched = False
            for ts in char_timestamps:
                if ts["char"] == expected_char:
                    aligned_result_list.append({
                        "char": expected_char,
                        "pinyin": pinyin_map[expected_char],
                        "start": ts["start"],
                        "end": ts["end"],
                        "score": ts.get("score", 1.0)
                    })
                    matched = True
                    break
            
            # If not matched, add placeholder with zero timestamps
            if not matched:
                aligned_result_list.append({
                    "char": expected_char,
                    "pinyin": pinyin_map[expected_char],
                    "start": 0.0,
                    "end": 0.0,
                    "score": 0.0
                })
        
        return aligned_result_list
    
    def is_available(self) -> bool:
        """Check if WhisperX is available."""
        return self.whisperx_available
