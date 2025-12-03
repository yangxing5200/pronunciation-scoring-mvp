"""
Task 3: Audio Slicing

Extracts audio segments for each character based on alignment timestamps.
Prepares data for WavLM embedding extraction.
"""

from typing import List, Dict
import numpy as np
from pathlib import Path


class AudioSlicer:
    """
    Slices audio into character-level segments based on alignment results.
    
    Outputs numpy arrays suitable for WavLM feature extraction.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio slicer.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def slice_audio(
        self,
        audio_path: str,
        alignment_results: List[Dict]
    ) -> List[Dict]:
        """
        Slice audio into character-level segments.
        
        Args:
            audio_path: Path to audio file
            alignment_results: List from ChineseAudioAligner with timestamps
        
        Returns:
            List of dictionaries with character info and audio segment:
            [
                {
                    "char": "你",
                    "pinyin": "ni3",
                    "start": 0.12,
                    "end": 0.36,
                    "audio_segment": np.array([...])  # Audio samples
                },
                ...
            ]
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError("librosa not available. Please install: pip install librosa")
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        sliced_results = []
        
        for item in alignment_results:
            char = item["char"]
            pinyin = item["pinyin"]
            start_time = item["start"]
            end_time = item["end"]
            
            # Convert time to sample indices
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, min(start_sample, len(audio)))
            end_sample = max(0, min(end_sample, len(audio)))
            
            # Extract audio segment
            if end_sample > start_sample:
                audio_segment = audio[start_sample:end_sample]
            else:
                # Empty segment if timestamps are invalid
                audio_segment = np.array([])
            
            sliced_results.append({
                "char": char,
                "pinyin": pinyin,
                "start": start_time,
                "end": end_time,
                "audio_segment": audio_segment,
                "duration": end_time - start_time
            })
        
        return sliced_results
    
    def save_segments(
        self,
        sliced_results: List[Dict],
        output_dir: str
    ) -> List[str]:
        """
        Save audio segments to individual files.
        
        Args:
            sliced_results: Output from slice_audio()
            output_dir: Directory to save segment files
        
        Returns:
            List of saved file paths
        """
        try:
            import soundfile as sf
        except ImportError:
            raise RuntimeError("soundfile not available. Please install: pip install soundfile")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, item in enumerate(sliced_results):
            char = item["char"]
            audio_segment = item["audio_segment"]
            
            if len(audio_segment) > 0:
                # Create filename
                filename = f"char_{i:03d}_{char}.wav"
                filepath = output_path / filename
                
                # Save audio segment
                sf.write(
                    str(filepath),
                    audio_segment,
                    self.sample_rate
                )
                
                saved_paths.append(str(filepath))
            else:
                saved_paths.append(None)
        
        return saved_paths
