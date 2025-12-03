"""
Task 6: Duration Scoring

Evaluates whether each character's pronunciation duration is reasonable.
Compares with expected duration ranges or reference audio.
"""

from typing import List, Dict, Optional
import numpy as np


class DurationScorer:
    """
    Duration scoring for Chinese character pronunciation.
    
    Evaluates if characters are pronounced with appropriate duration.
    """
    
    def __init__(self):
        """Initialize duration scorer."""
        # Typical duration ranges for Chinese characters (in seconds)
        # Based on empirical observations of native speakers
        self.typical_duration_min = 0.1  # 100ms
        self.typical_duration_max = 0.6  # 600ms
        self.optimal_duration = 0.25     # 250ms (average)
    
    def score_durations(
        self,
        sliced_results: List[Dict],
        reference_durations: Optional[List[float]] = None
    ) -> List[Dict]:
        """
        Score pronunciation duration for each character.
        
        Args:
            sliced_results: Output from AudioSlicer with timing info
            reference_durations: Optional list of reference durations
        
        Returns:
            List with duration scores added:
            [
                {"char":"你", "duration_score":0.98, ...},
                {"char":"好", "duration_score":0.85, ...},
                ...
            ]
        """
        scored_results = []
        
        for i, item in enumerate(sliced_results):
            duration = item.get("duration", 0.0)
            
            if reference_durations and i < len(reference_durations):
                # Compare with reference duration
                ref_duration = reference_durations[i]
                duration_score = self._score_with_reference(duration, ref_duration)
            else:
                # Score based on typical duration range
                duration_score = self._score_without_reference(duration)
            
            # Add to result
            result = item.copy()
            result["duration_score"] = float(duration_score)
            scored_results.append(result)
        
        return scored_results
    
    def _score_with_reference(
        self,
        user_duration: float,
        ref_duration: float
    ) -> float:
        """
        Score duration by comparing with reference.
        
        Args:
            user_duration: User's pronunciation duration
            ref_duration: Reference pronunciation duration
        
        Returns:
            Duration score (0.0 to 1.0)
        """
        if ref_duration <= 0:
            return 0.5  # Neutral score if no valid reference
        
        # Calculate relative difference
        ratio = user_duration / ref_duration
        
        # Ideal ratio is 1.0 (same duration)
        # Allow 20% variation for good score
        # Allow 50% variation for acceptable score
        
        if 0.8 <= ratio <= 1.2:
            # Within 20% - excellent
            score = 1.0
        elif 0.6 <= ratio <= 1.5:
            # Within 50% - good
            deviation = abs(ratio - 1.0)
            score = 1.0 - (deviation - 0.2) / 0.3 * 0.2
        elif 0.4 <= ratio <= 2.0:
            # Within 100% - acceptable
            deviation = abs(ratio - 1.0)
            score = 0.8 - (deviation - 0.5) / 1.0 * 0.4
        else:
            # Too different
            score = 0.4
        
        return max(0.0, min(1.0, score))
    
    def _score_without_reference(self, duration: float) -> float:
        """
        Score duration based on typical ranges.
        
        Args:
            duration: Pronunciation duration in seconds
        
        Returns:
            Duration score (0.0 to 1.0)
        """
        if duration <= 0:
            return 0.0
        
        # Check if within typical range
        if self.typical_duration_min <= duration <= self.typical_duration_max:
            # Within acceptable range
            # Best score for optimal duration
            distance_from_optimal = abs(duration - self.optimal_duration)
            max_distance = self.typical_duration_max - self.optimal_duration
            
            score = 1.0 - (distance_from_optimal / max_distance) * 0.2
            return max(0.8, min(1.0, score))
        
        elif duration < self.typical_duration_min:
            # Too short - penalize based on how short
            if duration < self.typical_duration_min / 2:
                return 0.3  # Very short
            else:
                ratio = duration / self.typical_duration_min
                return 0.3 + 0.5 * ratio  # 0.3 to 0.8
        
        else:
            # Too long - penalize based on how long
            if duration > self.typical_duration_max * 2:
                return 0.3  # Very long
            else:
                ratio = self.typical_duration_max / duration
                return 0.3 + 0.5 * ratio  # 0.3 to 0.8
    
    def extract_reference_durations(
        self,
        reference_audio_path: str,
        alignment_results: List[Dict]
    ) -> List[float]:
        """
        Extract reference durations from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio
            alignment_results: Alignment results with timestamps
        
        Returns:
            List of reference durations
        """
        # This would use the same alignment process on reference audio
        # For simplicity, use the durations from alignment if available
        durations = []
        
        for item in alignment_results:
            duration = item.get("end", 0.0) - item.get("start", 0.0)
            durations.append(duration)
        
        return durations
