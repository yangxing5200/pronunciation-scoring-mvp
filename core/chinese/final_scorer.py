"""
Task 9: Final Scoring

Integrates all sub-scores into a comprehensive final score.
Uses weighted formula to calculate overall pronunciation quality.
"""

from typing import List, Dict
import numpy as np


class FinalScorer:
    """
    Final comprehensive scoring for Chinese pronunciation.
    
    Combines scores from:
    - Acoustic scoring (50%)
    - Tone scoring (25%)
    - Duration scoring (15%)
    - Pause/fluency scoring (10%)
    """
    
    def __init__(
        self,
        acoustic_weight: float = 0.50,
        tone_weight: float = 0.25,
        duration_weight: float = 0.15,
        pause_weight: float = 0.10
    ):
        """
        Initialize final scorer.
        
        Args:
            acoustic_weight: Weight for acoustic score
            tone_weight: Weight for tone score
            duration_weight: Weight for duration score
            pause_weight: Weight for pause score
        """
        self.acoustic_weight = acoustic_weight
        self.tone_weight = tone_weight
        self.duration_weight = duration_weight
        self.pause_weight = pause_weight
        
        # Validate weights sum to 1.0
        total_weight = acoustic_weight + tone_weight + duration_weight + pause_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def calculate_final_scores(
        self,
        scored_results: List[Dict]
    ) -> List[Dict]:
        """
        Calculate final scores for each character.
        
        Args:
            scored_results: Results with all sub-scores
        
        Returns:
            List with final scores added:
            [
                {"char":"你", "final_score":94, ...},
                {"char":"好", "final_score":91, ...},
                ...
            ]
        """
        final_results = []
        
        for item in scored_results:
            # Extract sub-scores (default to 0.7 if missing)
            acoustic_score = item.get("acoustic_score", 0.7)
            tone_score = item.get("tone_score", 0.7)
            duration_score = item.get("duration_score", 0.7)
            pause_score = item.get("pause_score", 0.7)
            
            # Calculate weighted final score
            final_score = (
                self.acoustic_weight * acoustic_score +
                self.tone_weight * tone_score +
                self.duration_weight * duration_score +
                self.pause_weight * pause_score
            )
            
            # Convert to 0-100 scale
            final_score_100 = int(final_score * 100)
            
            # Add to result
            result = item.copy()
            result["final_score"] = final_score_100
            final_results.append(result)
        
        return final_results
    
    def calculate_overall_score(
        self,
        final_results: List[Dict]
    ) -> Dict:
        """
        Calculate overall pronunciation score for the entire utterance.
        
        Args:
            final_results: Results with final scores
        
        Returns:
            Dictionary with overall metrics
        """
        if len(final_results) == 0:
            return {
                "overall_score": 0,
                "avg_acoustic_score": 0,
                "avg_tone_score": 0,
                "avg_duration_score": 0,
                "avg_pause_score": 0,
                "num_characters": 0
            }
        
        # Calculate averages
        final_scores = [item.get("final_score", 0) for item in final_results]
        acoustic_scores = [item.get("acoustic_score", 0) * 100 for item in final_results]
        tone_scores = [item.get("tone_score", 0) * 100 for item in final_results]
        duration_scores = [item.get("duration_score", 0) * 100 for item in final_results]
        pause_scores = [item.get("pause_score", 0) * 100 for item in final_results]
        
        overall_score = int(np.mean(final_scores))
        
        return {
            "overall_score": overall_score,
            "avg_acoustic_score": int(np.mean(acoustic_scores)),
            "avg_tone_score": int(np.mean(tone_scores)),
            "avg_duration_score": int(np.mean(duration_scores)),
            "avg_pause_score": int(np.mean(pause_scores)),
            "num_characters": len(final_results),
            "min_score": int(np.min(final_scores)),
            "max_score": int(np.max(final_scores))
        }
    
    def generate_feedback(
        self,
        final_results: List[Dict],
        overall_metrics: Dict
    ) -> List[str]:
        """
        Generate human-readable feedback based on scores.
        
        Args:
            final_results: Results with final scores
            overall_metrics: Overall metrics
        
        Returns:
            List of feedback messages
        """
        feedback = []
        
        overall_score = overall_metrics["overall_score"]
        
        # Overall performance feedback
        if overall_score >= 90:
            feedback.append("优秀！发音非常标准。(Excellent! Very standard pronunciation.)")
        elif overall_score >= 75:
            feedback.append("良好！发音较为标准，继续保持。(Good! Pronunciation is quite standard.)")
        elif overall_score >= 60:
            feedback.append("及格。发音需要继续练习。(Pass. Pronunciation needs more practice.)")
        else:
            feedback.append("需要改进。建议多听多练。(Needs improvement. Listen and practice more.)")
        
        # Find problematic characters
        low_score_chars = [
            item for item in final_results
            if item.get("final_score", 0) < 70
        ]
        
        if low_score_chars:
            low_score_chars.sort(key=lambda x: x.get("final_score", 0))
            char_list = "、".join([item["char"] for item in low_score_chars[:3]])
            feedback.append(f"重点改进字：{char_list}")
        
        # Specific dimension feedback
        avg_acoustic = overall_metrics["avg_acoustic_score"]
        avg_tone = overall_metrics["avg_tone_score"]
        avg_duration = overall_metrics["avg_duration_score"]
        avg_pause = overall_metrics["avg_pause_score"]
        
        if avg_acoustic < 70:
            feedback.append("声母韵母发音需要加强。(Initial/final pronunciation needs improvement.)")
        
        if avg_tone < 70:
            feedback.append("声调需要多加练习。(Tone needs more practice.)")
        
        if avg_duration < 70:
            feedback.append("发音时长控制需要改善。(Duration control needs improvement.)")
        
        if avg_pause < 70:
            feedback.append("语音流畅度需要提高，注意减少停顿。(Fluency needs improvement, reduce pauses.)")
        
        return feedback
