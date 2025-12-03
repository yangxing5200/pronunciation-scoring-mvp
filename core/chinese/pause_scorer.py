"""
Task 7: Pause/Fluency Scoring

Evaluates fluency by analyzing pauses between characters.
Detects unnatural pauses and choppy pronunciation.
"""

from typing import List, Dict
import numpy as np


class PauseScorer:
    """
    Pause/fluency scoring for Chinese pronunciation.
    
    Evaluates speaking fluency by analyzing inter-character pauses.
    """
    
    def __init__(self):
        """Initialize pause scorer."""
        # Pause thresholds (in seconds)
        self.natural_pause_max = 0.15     # Max natural pause (150ms)
        self.acceptable_pause_max = 0.3   # Max acceptable pause (300ms)
        self.long_pause_threshold = 0.5   # Long pause threshold (500ms)
    
    def score_pauses(
        self,
        alignment_results: List[Dict]
    ) -> List[Dict]:
        """
        Score fluency based on pauses between characters.
        
        Args:
            alignment_results: Alignment results with timestamps
        
        Returns:
            List with pause scores added:
            [
                {"char":"你", "pause_score":1.0, "pause_after":0.05, ...},
                {"char":"好", "pause_score":0.95, "pause_after":0.10, ...},
                ...
            ]
        """
        if len(alignment_results) == 0:
            return []
        
        scored_results = []
        
        for i, item in enumerate(alignment_results):
            # Calculate pause after this character
            if i < len(alignment_results) - 1:
                current_end = item.get("end", 0.0)
                next_start = alignment_results[i + 1].get("start", 0.0)
                pause_after = max(0.0, next_start - current_end)
            else:
                # Last character - no pause after
                pause_after = 0.0
            
            # Score the pause
            pause_score = self._score_pause(pause_after)
            
            # Add to result
            result = item.copy()
            result["pause_score"] = float(pause_score)
            result["pause_after"] = float(pause_after)
            scored_results.append(result)
        
        return scored_results
    
    def _score_pause(self, pause_duration: float) -> float:
        """
        Score a single pause.
        
        Args:
            pause_duration: Duration of pause in seconds
        
        Returns:
            Pause score (0.0 to 1.0)
        """
        if pause_duration < 0:
            # Overlapping - might indicate rushed speech
            return 0.9
        
        elif pause_duration <= self.natural_pause_max:
            # Natural pause - excellent
            return 1.0
        
        elif pause_duration <= self.acceptable_pause_max:
            # Acceptable pause - good
            excess = pause_duration - self.natural_pause_max
            penalty = excess / (self.acceptable_pause_max - self.natural_pause_max) * 0.1
            return 1.0 - penalty
        
        elif pause_duration <= self.long_pause_threshold:
            # Long pause - fair
            excess = pause_duration - self.acceptable_pause_max
            penalty = 0.1 + excess / (self.long_pause_threshold - self.acceptable_pause_max) * 0.3
            return 1.0 - penalty
        
        else:
            # Very long pause - poor fluency
            excess = pause_duration - self.long_pause_threshold
            penalty = 0.4 + min(0.5, excess * 0.5)
            return max(0.1, 1.0 - penalty)
    
    def calculate_overall_fluency(
        self,
        scored_results: List[Dict]
    ) -> Dict:
        """
        Calculate overall fluency metrics.
        
        Args:
            scored_results: Results with pause scores
        
        Returns:
            Dictionary with overall fluency metrics
        """
        if len(scored_results) == 0:
            return {
                "overall_fluency_score": 0.0,
                "avg_pause": 0.0,
                "max_pause": 0.0,
                "num_long_pauses": 0
            }
        
        pause_scores = [item.get("pause_score", 1.0) for item in scored_results]
        pauses = [item.get("pause_after", 0.0) for item in scored_results if "pause_after" in item]
        
        # Overall fluency is average of pause scores
        overall_score = np.mean(pause_scores)
        
        # Calculate statistics
        avg_pause = np.mean(pauses) if pauses else 0.0
        max_pause = np.max(pauses) if pauses else 0.0
        num_long_pauses = sum(1 for p in pauses if p > self.long_pause_threshold)
        
        return {
            "overall_fluency_score": float(overall_score),
            "avg_pause": float(avg_pause),
            "max_pause": float(max_pause),
            "num_long_pauses": num_long_pauses
        }
