"""
Comprehensive pronunciation scoring system.
"""

import numpy as np
import string
from typing import Dict, List, Optional
from .aligner import PhonemeAligner
from .text_comparator import TextComparator


class PronunciationScorer:
    """
    Three-dimensional pronunciation scorer: Accuracy, Fluency, Prosody.
    """
    
    def __init__(self):
        """Initialize pronunciation scorer."""
        self.aligner = PhonemeAligner()
        self.comparator = TextComparator()
        
        # Phoneme similarity matrix (simplified)
        # TODO: Load from configuration file or use standardized phoneme distance matrix
        # Current values are empirical approximations
        # Format: (phoneme1, phoneme2): similarity_score (0.0-1.0)
        self.phoneme_similarities = {
            ('i', 'ɪ'): 0.7,   # Close vowels
            ('e', 'ɛ'): 0.7,   # Mid vowels
            ('æ', 'ɛ'): 0.6,   # Near-open vs mid vowels
            ('s', 'ʃ'): 0.5,   # Sibilants
            ('r', 'l'): 0.3,   # Liquids (often confused)
            ('θ', 's'): 0.5,   # Fricatives
            ('ð', 'z'): 0.5,   # Voiced fricatives
        }
    
    def _remove_punctuation(self, text: str) -> str:
        """
        Remove punctuation from text for better word matching.
        
        Args:
            text: Input text
        
        Returns:
            Text with punctuation removed
        """
        # Remove common punctuation marks
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def score_pronunciation(
        self,
        user_audio_path: str,
        reference_text: str,
        transcribed_text: str,
        word_timestamps: List[Dict],
        reference_audio_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive pronunciation score.
        
        Args:
            user_audio_path: Path to user's audio recording
            reference_text: Expected text
            transcribed_text: What was transcribed from user audio
            word_timestamps: Word-level timestamps from transcription
            reference_audio_path: Optional path to reference audio
        
        Returns:
            Comprehensive scoring dictionary
        """
        # 1. Text comparison (accuracy baseline)
        text_comparison = self.comparator.compare_texts(
            reference_text,
            transcribed_text
        )
        
        # 2. Word-level scoring
        word_scores = self._score_words(
            reference_text,
            transcribed_text,
            word_timestamps
        )
        
        # 3. Fluency analysis
        fluency_score = self._analyze_fluency(word_timestamps)
        
        # 4. Acoustic/Prosody analysis
        prosody_score = 70  # Default
        if reference_audio_path:
            prosody_score = self._analyze_prosody(
                user_audio_path,
                reference_audio_path,
                word_timestamps
            )
        
        # 5. Overall accuracy score
        accuracy_score = self._calculate_accuracy(
            text_comparison,
            word_scores
        )
        
        # 6. Calculate total score (weighted average)
        total_score = int(
            0.5 * accuracy_score +
            0.25 * fluency_score +
            0.25 * prosody_score
        )
        
        # 7. Identify main issues
        issues = self._identify_issues(
            word_scores,
            fluency_score,
            prosody_score,
            text_comparison
        )
        
        # 8. Generate phoneme-level scores (simplified)
        phoneme_scores = self._generate_phoneme_scores(word_scores)
        
        return {
            "total_score": total_score,
            "accuracy": accuracy_score,
            "fluency": fluency_score,
            "prosody": prosody_score,
            "word_scores": word_scores,
            "phoneme_scores": phoneme_scores,
            "issues": issues[:3],  # Top 3 issues
            "text_comparison": text_comparison
        }
    
    def _score_words(
        self,
        reference_text: str,
        transcribed_text: str,
        word_timestamps: List[Dict]
    ) -> List[Dict]:
        """
        Score individual words.
        
        Args:
            reference_text: Expected text
            transcribed_text: Transcribed text
            word_timestamps: Word timestamps
        
        Returns:
            List of word scores
        """
        # Remove punctuation before splitting to avoid matching issues
        ref_clean = self._remove_punctuation(reference_text.lower())
        trans_clean = self._remove_punctuation(transcribed_text.lower())
        
        ref_words = ref_clean.split()
        trans_words = trans_clean.split()
        
        word_scores = []
        
        # Track which transcribed words have been matched
        matched_indices = set()
        
        # Align words using improved matching algorithm
        for i, ref_word in enumerate(ref_words):
            score = 0
            status = "missing"
            
            # First, try exact match in the transcription
            if ref_word in trans_words:
                # Find the closest position match that hasn't been used
                best_match_idx = None
                for idx, trans_word in enumerate(trans_words):
                    if trans_word == ref_word and idx not in matched_indices:
                        # Prefer matches close to the expected position
                        if best_match_idx is None or abs(idx - i) < abs(best_match_idx - i):
                            best_match_idx = idx
                
                if best_match_idx is not None:
                    matched_indices.add(best_match_idx)
                    score = 90
                    status = "correct"
            
            # If no exact match, try positional comparison
            if score == 0 and i < len(trans_words):
                # Check similarity with word at same position
                similarity = self.comparator.calculate_word_similarity(
                    ref_word,
                    trans_words[i]
                )
                score = int(similarity * 100)
                status = "partial" if score > 50 else "incorrect"
            
            # If still no good match, try finding best match in remaining words
            if score < 50:
                remaining_words = [
                    trans_words[idx] for idx in range(len(trans_words))
                    if idx not in matched_indices
                ]
                if remaining_words:
                    best_match, best_similarity = self.comparator.find_closest_match(
                        ref_word,
                        remaining_words
                    )
                    if best_similarity > score / 100:
                        # Update score if we found a better match
                        score = int(best_similarity * 100)
                        status = "partial" if score > 50 else "incorrect"
                        # Mark this word as matched
                        for idx, trans_word in enumerate(trans_words):
                            if trans_word == best_match and idx not in matched_indices:
                                matched_indices.add(idx)
                                break
            
            word_scores.append({
                "word": ref_words[i],  # Use cleaned word
                "score": score,
                "status": status
            })
        
        return word_scores
    
    def _analyze_fluency(self, word_timestamps: List[Dict]) -> int:
        """
        Analyze fluency based on pauses and rhythm.
        
        Args:
            word_timestamps: Word-level timestamps
        
        Returns:
            Fluency score (0-100)
        """
        if len(word_timestamps) < 2:
            return 85  # Default for very short utterances
        
        # Calculate inter-word pauses
        pauses = []
        for i in range(len(word_timestamps) - 1):
            gap = word_timestamps[i + 1]["start"] - word_timestamps[i]["end"]
            pauses.append(gap)
        
        # Penalize long pauses
        avg_pause = np.mean(pauses)
        max_pause = np.max(pauses)
        
        score = 100
        
        # Penalty for average pause length
        if avg_pause > 0.5:
            score -= min(20, int((avg_pause - 0.5) * 20))
        
        # Penalty for very long pauses
        if max_pause > 1.0:
            score -= min(15, int((max_pause - 1.0) * 10))
        
        # Penalty for very short pauses (speaking too fast)
        if avg_pause < 0.05:
            score -= 10
        
        return max(0, score)
    
    def _analyze_prosody(
        self,
        user_audio_path: str,
        reference_audio_path: str,
        word_timestamps: List[Dict]
    ) -> int:
        """
        Analyze prosody (pitch, energy, rhythm).
        
        Args:
            user_audio_path: Path to user audio
            reference_audio_path: Path to reference audio
            word_timestamps: Word timestamps
        
        Returns:
            Prosody score (0-100)
        """
        # Perform DTW alignment
        alignment = self.aligner.align_with_dtw(
            user_audio_path,
            reference_audio_path
        )
        
        # Lower distance means better match
        normalized_distance = alignment["normalized_distance"]
        
        # Convert distance to score (empirical mapping)
        # Typical normalized DTW distance ranges from 1-10
        if normalized_distance < 2.0:
            score = 95
        elif normalized_distance < 3.0:
            score = 85
        elif normalized_distance < 4.0:
            score = 75
        elif normalized_distance < 5.0:
            score = 65
        elif normalized_distance < 6.0:
            score = 55
        else:
            score = 45
        
        return score
    
    def _calculate_accuracy(
        self,
        text_comparison: Dict,
        word_scores: List[Dict]
    ) -> int:
        """
        Calculate overall pronunciation accuracy.
        
        Args:
            text_comparison: Text comparison results
            word_scores: Word-level scores
        
        Returns:
            Accuracy score (0-100)
        """
        if not word_scores:
            return 0
        
        # Average word scores
        avg_word_score = np.mean([w["score"] for w in word_scores])
        
        # Factor in WER (Word Error Rate)
        wer = text_comparison.get("wer", 0.0)
        wer_penalty = wer * 30  # WER of 1.0 gives 30 point penalty
        
        accuracy = int(avg_word_score - wer_penalty)
        
        return max(0, min(100, accuracy))
    
    def _identify_issues(
        self,
        word_scores: List[Dict],
        fluency_score: int,
        prosody_score: int,
        text_comparison: Dict
    ) -> List[str]:
        """
        Identify top pronunciation issues.
        
        Args:
            word_scores: Word scores
            fluency_score: Fluency score
            prosody_score: Prosody score
            text_comparison: Text comparison results
        
        Returns:
            List of issue descriptions
        """
        issues = []
        
        # Find worst performing words
        low_score_words = [
            w for w in word_scores if w["score"] < 75
        ]
        low_score_words.sort(key=lambda x: x["score"])
        
        for word_info in low_score_words[:2]:
            word = word_info["word"]
            score = word_info["score"]
            if word_info["status"] == "missing":
                issues.append(f"Word '{word}' was not recognized - check pronunciation")
            elif word_info["status"] == "incorrect":
                issues.append(f"Word '{word}' pronunciation needs improvement (score: {score})")
        
        # Fluency issues
        if fluency_score < 75:
            issues.append("Speech rhythm is choppy - try to speak more smoothly")
        
        # Prosody issues
        if prosody_score < 70:
            issues.append("Pitch and intonation differ from native speaker pattern")
        
        # Missing/extra words
        if text_comparison.get("missing_words"):
            missing = text_comparison["missing_words"][:2]
            issues.append(f"Missing words: {', '.join(missing)}")
        
        if text_comparison.get("extra_words"):
            extra = text_comparison["extra_words"][:2]
            issues.append(f"Extra words detected: {', '.join(extra)}")
        
        return issues
    
    def _generate_phoneme_scores(
        self,
        word_scores: List[Dict]
    ) -> List[Dict]:
        """
        Generate phoneme-level scores (simplified).
        
        Args:
            word_scores: Word-level scores
        
        Returns:
            List of phoneme scores
        """
        phoneme_scores = []
        
        for word_info in word_scores:
            word = word_info["word"]
            score = word_info["score"]
            
            # Simplified: treat each character as a phoneme representation
            for i, char in enumerate(word):
                phoneme_scores.append({
                    "char": char,
                    "word": word,
                    "phoneme": char,  # Simplified
                    "score": score,
                    "status": word_info["status"]
                })
        
        return phoneme_scores
