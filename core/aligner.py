"""
Phoneme alignment using Dynamic Time Warping (DTW).
"""

import numpy as np
import librosa
from dtw import dtw
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PhonemeAligner:
    """
    Aligner for phoneme-level analysis using DTW and acoustic features.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize phoneme aligner.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if needed.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio, sr
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal
            n_mfcc: Number of MFCC coefficients
        
        Returns:
            MFCC feature matrix (n_mfcc, time_frames)
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )
        # Normalize
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (
            np.std(mfcc, axis=1, keepdims=True) + 1e-8
        )
        return mfcc
    
    def extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract pitch (F0) from audio using librosa.
        
        Args:
            audio: Audio signal
        
        Returns:
            Pitch contour (time_frames,)
        """
        # Extract pitch using pyin algorithm
        # Note: voiced_flag and voiced_probs could be used for more advanced
        # voicing detection, but for this MVP we only use the F0 values
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0
    
    def extract_energy(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract energy (RMS) from audio.
        
        Args:
            audio: Audio signal
        
        Returns:
            Energy contour (time_frames,)
        """
        energy = librosa.feature.rms(y=audio)[0]
        return energy
    
    def align_with_dtw(
        self,
        user_audio_path: str,
        reference_audio_path: str
    ) -> Dict:
        """
        Align user audio with reference audio using DTW.
        
        Args:
            user_audio_path: Path to user's audio
            reference_audio_path: Path to reference audio
        
        Returns:
            Dictionary containing alignment results
        """
        # Load audio files
        user_audio, _ = self.load_audio(user_audio_path)
        ref_audio, _ = self.load_audio(reference_audio_path)
        
        # Extract features
        user_mfcc = self.extract_mfcc(user_audio)
        ref_mfcc = self.extract_mfcc(ref_audio)
        
        # Perform DTW alignment
        distance, cost_matrix, acc_cost_matrix, path = dtw(
            user_mfcc.T,
            ref_mfcc.T,
            dist=lambda x, y: np.linalg.norm(x - y)
        )
        
        # Normalize distance by path length
        normalized_distance = distance / len(path[0])
        
        return {
            "distance": distance,
            "normalized_distance": normalized_distance,
            "path": path,
            "cost_matrix": cost_matrix,
            "user_mfcc": user_mfcc,
            "ref_mfcc": ref_mfcc
        }
    
    def segment_phonemes(
        self,
        audio_path: str,
        word_timestamps: List[Dict],
        phoneme_map: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Segment audio into phonemes based on word timestamps.
        
        Args:
            audio_path: Path to audio file
            word_timestamps: List of word timestamps from transcription
            phoneme_map: Optional mapping of words to phonemes
        
        Returns:
            List of phoneme segments with timestamps
        """
        audio, sr = self.load_audio(audio_path)
        
        # Extract acoustic features
        pitch = self.extract_pitch(audio)
        energy = self.extract_energy(audio)
        
        # Calculate frame times for pitch and energy
        hop_length = 512  # Default librosa hop length
        frame_times = librosa.frames_to_time(
            np.arange(len(pitch)),
            sr=sr,
            hop_length=hop_length
        )
        
        phoneme_segments = []
        
        for word_info in word_timestamps:
            word = word_info.get("word", "")
            start_time = word_info.get("start", 0.0)
            end_time = word_info.get("end", 0.0)
            
            # Find corresponding frames
            start_idx = np.argmin(np.abs(frame_times - start_time))
            end_idx = np.argmin(np.abs(frame_times - end_time))
            
            # Extract features for this segment
            segment_pitch = pitch[start_idx:end_idx]
            segment_energy = energy[start_idx:end_idx]
            
            # Calculate statistics
            avg_pitch = np.mean(segment_pitch[segment_pitch > 0]) if np.any(segment_pitch > 0) else 0
            avg_energy = np.mean(segment_energy)
            
            phoneme_segments.append({
                "word": word,
                "start": start_time,
                "end": end_time,
                "duration": end_time - start_time,
                "avg_pitch": float(avg_pitch),
                "avg_energy": float(avg_energy),
                "pitch_contour": segment_pitch.tolist(),
                "energy_contour": segment_energy.tolist()
            })
        
        return phoneme_segments
    
    def compare_acoustic_features(
        self,
        user_segment: Dict,
        reference_segment: Dict
    ) -> Dict:
        """
        Compare acoustic features between user and reference segments.
        
        Args:
            user_segment: User's phoneme segment
            reference_segment: Reference phoneme segment
        
        Returns:
            Dictionary with comparison metrics
        """
        # Pitch deviation
        user_pitch = user_segment.get("avg_pitch", 0)
        ref_pitch = reference_segment.get("avg_pitch", 0)
        
        if ref_pitch > 0:
            pitch_deviation = abs(user_pitch - ref_pitch) / ref_pitch
        else:
            pitch_deviation = 0.0
        
        # Energy deviation
        user_energy = user_segment.get("avg_energy", 0)
        ref_energy = reference_segment.get("avg_energy", 0)
        
        if ref_energy > 0:
            energy_deviation = abs(user_energy - ref_energy) / ref_energy
        else:
            energy_deviation = 0.0
        
        # Duration deviation
        user_duration = user_segment.get("duration", 0)
        ref_duration = reference_segment.get("duration", 0)
        
        if ref_duration > 0:
            duration_deviation = abs(user_duration - ref_duration) / ref_duration
        else:
            duration_deviation = 0.0
        
        return {
            "pitch_deviation": pitch_deviation,
            "energy_deviation": energy_deviation,
            "duration_deviation": duration_deviation,
            "user_pitch": user_pitch,
            "ref_pitch": ref_pitch,
            "user_energy": user_energy,
            "ref_energy": ref_energy
        }
