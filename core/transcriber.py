"""
Whisper-based speech transcription module for offline use.

Supports:
- Word-level timestamps (native Whisper)
- Token-level timestamps (native Whisper)
- Phoneme-level timestamps (optional WhisperX)
"""

import os
import warnings
import re
from typing import Dict, List, Optional, Union
import torch
import whisper
from pathlib import Path


class WhisperTranscriber:
    """
    Offline Whisper transcriber with word-level and token-level timestamps.
    
    Supports optional WhisperX for enhanced phoneme-level alignment.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        model_dir: str = "models/whisper",
        device: Optional[str] = None,
        language: str = "en",
        use_whisperx: bool = False
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: Size of Whisper model (tiny, base, small, medium, large)
            model_dir: Directory containing downloaded Whisper models
            device: Device to use (cuda/cpu), auto-detect if None
            language: Target language code
            use_whisperx: Use WhisperX for enhanced alignment (requires whisperx package)
        """
        self.model_size = model_size
        self.model_dir = Path(model_dir)
        self.language = language
        self.use_whisperx = use_whisperx
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.whisperx_model = None
        self.whisperx_align_model = None
        self.whisperx_metadata = None
        
        self._load_model()
        if self.use_whisperx:
            self._load_whisperx()
    
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
    
    def _load_whisperx(self):
        """Load WhisperX for enhanced alignment (optional)."""
        try:
            import whisperx
            
            # Load WhisperX model
            self.whisperx_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            # Load alignment model for the target language
            self.whisperx_align_model, self.whisperx_metadata = whisperx.load_align_model(
                language_code=self.language,
                device=self.device
            )
            
            print(f"WhisperX alignment model loaded for language: {self.language}")
            
        except ImportError:
            warnings.warn(
                "WhisperX not available. Install with: pip install whisperx\n"
                "Falling back to standard Whisper word-level timestamps."
            )
            self.use_whisperx = False
        except Exception as e:
            warnings.warn(f"Failed to load WhisperX: {e}\nUsing standard Whisper.")
            self.use_whisperx = False
    
    def _split_chinese_characters(self, text: str, start_time: float, end_time: float) -> List[Dict]:
        """
        Split Chinese text into individual characters with proportional timestamps.
        
        Args:
            text: Chinese text to split
            start_time: Start time of the text segment
            end_time: End time of the text segment
        
        Returns:
            List of dictionaries with word (character), start, end, probability
        """
        # Extract only Chinese characters (U+4E00 to U+9FFF)
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        if not chinese_chars:
            return []
        
        total_duration = end_time - start_time
        # Guard against division by zero (though check above prevents empty list)
        if len(chinese_chars) == 0:
            return []
        
        char_duration = total_duration / len(chinese_chars)
        
        result = []
        for i, char in enumerate(chinese_chars):
            result.append({
                'word': char,
                'start': start_time + i * char_duration,
                'end': start_time + (i + 1) * char_duration,
                'probability': 1.0
            })
        
        return result
    
    def _is_chinese(self, text: str) -> bool:
        """
        Detect if text contains Chinese characters.
        
        Args:
            text: Text to check
        
        Returns:
            True if text contains Chinese characters
        """
        return bool(re.search(r'[\u4e00-\u9fff]', text))
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio file with word-level timestamps.
        Uses WhisperX for enhanced alignment if available.
        
        Args:
            audio_path: Path to audio file
            language: Override language (uses self.language if None)
        
        Returns:
            Dictionary containing:
                - text: Full transcription
                - segments: List of segments with timestamps
                - words: List of word-level timestamps
                - language: Detected/specified language
                - alignment_type: 'whisperx' or 'whisper' indicating which was used
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        target_lang = language or self.language
        
        # Use WhisperX if available for better alignment
        if self.use_whisperx and self.whisperx_model is not None:
            return self._transcribe_with_whisperx(audio_path, target_lang)
        else:
            return self._transcribe_with_whisper(audio_path, target_lang)
    
    def _transcribe_with_whisper(
        self,
        audio_path: Path,
        target_lang: str
    ) -> Dict:
        """
        Transcribe using standard Whisper with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            target_lang: Target language code
        
        Returns:
            Transcription result dictionary
        """
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
            segment_words = segment.get("words", [])
            
            # Check if this is Chinese text
            segment_text = segment.get("text", "")
            if self._is_chinese(segment_text) and segment_words:
                # For Chinese, split each "word" into individual characters
                for word_info in segment_words:
                    word_text = word_info.get("word", "").strip()
                    word_start = word_info.get("start", 0.0)
                    word_end = word_info.get("end", 0.0)
                    
                    # Split this word into characters
                    char_words = self._split_chinese_characters(
                        word_text, word_start, word_end
                    )
                    words.extend(char_words)
            else:
                # For non-Chinese, use words as-is
                for word_info in segment_words:
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
            "phonemes": [],  # Standard Whisper doesn't provide phoneme-level alignment
            "language": result.get("language", target_lang),
            "alignment_type": "whisper"
        }
    
    def _transcribe_with_whisperx(
        self,
        audio_path: Path,
        target_lang: str
    ) -> Dict:
        """
        Transcribe using WhisperX for enhanced word-level and phoneme-level alignment.
        
        For English: Provides both word-level and phoneme-level alignment
        For Chinese: Provides enhanced word-level (character-level) alignment
        
        Args:
            audio_path: Path to audio file
            target_lang: Target language code
        
        Returns:
            Transcription result dictionary with enhanced alignment
        """
        try:
            import whisperx
            
            # Load audio
            audio = whisperx.load_audio(str(audio_path))
            
            # Transcribe with WhisperX
            result = self.whisperx_model.transcribe(
                audio,
                batch_size=16,
                language=target_lang
            )
            
            # Align whisper output for better timestamps
            # For English: get phoneme-level alignment
            # For Chinese: get character-level alignment
            return_char_alignments = self._is_chinese_lang(target_lang)
            
            if self.whisperx_align_model is not None:
                result = whisperx.align(
                    result["segments"],
                    self.whisperx_align_model,
                    self.whisperx_metadata,
                    audio,
                    self.device,
                    return_char_alignments=return_char_alignments
                )
            
            # Extract word-level and phoneme-level information
            words = []
            phonemes = []  # For English phoneme-level alignment
            full_text = []
            
            for segment in result.get("segments", []):
                segment_text = segment.get("text", "")
                full_text.append(segment_text)
                
                # WhisperX provides word-level alignment
                segment_words = segment.get("words", [])
                
                # Check if this is Chinese text
                if self._is_chinese(segment_text) and segment_words:
                    # For Chinese, use character alignments if available
                    for word_info in segment_words:
                        word_text = word_info.get("word", "").strip()
                        word_start = word_info.get("start", 0.0)
                        word_end = word_info.get("end", 0.0)
                        
                        # If character alignments are available, use them
                        if "chars" in word_info:
                            for char_info in word_info["chars"]:
                                words.append({
                                    "word": char_info.get("char", "").strip(),
                                    "start": char_info.get("start", word_start),
                                    "end": char_info.get("end", word_end),
                                    "probability": char_info.get("score", 1.0)
                                })
                        else:
                            # Fallback to proportional splitting
                            char_words = self._split_chinese_characters(
                                word_text, word_start, word_end
                            )
                            words.extend(char_words)
                else:
                    # For English and other languages, use word alignments
                    for word_info in segment_words:
                        word_text = word_info.get("word", "").strip()
                        word_start = word_info.get("start", 0.0)
                        word_end = word_info.get("end", 0.0)
                        
                        words.append({
                            "word": word_text,
                            "start": word_start,
                            "end": word_end,
                            "probability": word_info.get("score", 1.0)
                        })
                        
                        # Extract phoneme-level alignment for English
                        # WhisperX provides phoneme info in the alignment
                        if "phones" in word_info:
                            for phone_info in word_info["phones"]:
                                phonemes.append({
                                    "phoneme": phone_info.get("phone", ""),
                                    "word": word_text,
                                    "start": phone_info.get("start", word_start),
                                    "end": phone_info.get("end", word_end),
                                    "probability": phone_info.get("score", 1.0)
                                })
            
            return {
                "text": " ".join(full_text).strip(),
                "segments": result.get("segments", []),
                "words": words,
                "phonemes": phonemes,  # Phoneme-level alignment (mainly for English)
                "language": target_lang,
                "alignment_type": "whisperx"
            }
            
        except Exception as e:
            warnings.warn(f"WhisperX transcription failed: {e}\nFalling back to Whisper.")
            return self._transcribe_with_whisper(audio_path, target_lang)
    
    def _is_chinese_lang(self, lang_code: str) -> bool:
        """
        Check if language code is Chinese.
        
        Args:
            lang_code: Language code (e.g., 'zh', 'zh-CN', 'zh-TW')
        
        Returns:
            True if language is Chinese
        """
        return lang_code.lower().startswith('zh') or lang_code.lower() == 'chinese'
    
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
