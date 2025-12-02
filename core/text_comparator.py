"""
Text comparison module using Levenshtein distance and WER.
"""

import Levenshtein
from jiwer import wer, cer
from typing import Dict, List, Tuple


class TextComparator:
    """
    Compare transcribed text with reference text.
    """
    
    def __init__(self):
        """Initialize text comparator."""
        pass
    
    def compare_texts(
        self,
        reference: str,
        hypothesis: str
    ) -> Dict:
        """
        Compare reference and hypothesis texts.
        
        Args:
            reference: Expected/reference text
            hypothesis: Transcribed/hypothesis text
        
        Returns:
            Dictionary with comparison metrics
        """
        # Normalize texts
        ref_normalized = reference.lower().strip()
        hyp_normalized = hypothesis.lower().strip()
        
        # Calculate metrics
        wer_score = wer(ref_normalized, hyp_normalized) if hyp_normalized else 1.0
        cer_score = cer(ref_normalized, hyp_normalized) if hyp_normalized else 1.0
        
        # Levenshtein distance
        lev_distance = Levenshtein.distance(ref_normalized, hyp_normalized)
        
        # Similarity ratio
        similarity = Levenshtein.ratio(ref_normalized, hyp_normalized)
        
        # Word-level analysis
        ref_words = ref_normalized.split()
        hyp_words = hyp_normalized.split()
        
        missing_words, extra_words, correct_words = self._analyze_word_differences(
            ref_words,
            hyp_words
        )
        
        return {
            "wer": wer_score,
            "cer": cer_score,
            "levenshtein_distance": lev_distance,
            "similarity": similarity,
            "reference": ref_normalized,
            "hypothesis": hyp_normalized,
            "missing_words": missing_words,
            "extra_words": extra_words,
            "correct_words": correct_words,
            "match": ref_normalized == hyp_normalized
        }
    
    def _analyze_word_differences(
        self,
        reference_words: List[str],
        hypothesis_words: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyze word-level differences.
        
        Args:
            reference_words: List of reference words
            hypothesis_words: List of hypothesis words
        
        Returns:
            Tuple of (missing_words, extra_words, correct_words)
        """
        ref_set = set(reference_words)
        hyp_set = set(hypothesis_words)
        
        missing_words = list(ref_set - hyp_set)
        extra_words = list(hyp_set - ref_set)
        correct_words = list(ref_set & hyp_set)
        
        return missing_words, extra_words, correct_words
    
    def calculate_word_similarity(
        self,
        word1: str,
        word2: str
    ) -> float:
        """
        Calculate similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        return Levenshtein.ratio(word1.lower(), word2.lower())
    
    def find_closest_match(
        self,
        word: str,
        candidates: List[str]
    ) -> Tuple[str, float]:
        """
        Find closest matching word from candidates.
        
        Args:
            word: Word to match
            candidates: List of candidate words
        
        Returns:
            Tuple of (best_match, similarity_score)
        """
        if not candidates:
            return "", 0.0
        
        best_match = ""
        best_score = 0.0
        
        for candidate in candidates:
            score = self.calculate_word_similarity(word, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        return best_match, best_score
    
    def detect_word_order_errors(
        self,
        reference_words: List[str],
        hypothesis_words: List[str]
    ) -> List[Dict]:
        """
        Detect word order errors.
        
        Args:
            reference_words: Reference word list
            hypothesis_words: Hypothesis word list
        
        Returns:
            List of order error descriptions
        """
        errors = []
        
        # Simple position-based check
        min_len = min(len(reference_words), len(hypothesis_words))
        
        for i in range(min_len):
            if reference_words[i] != hypothesis_words[i]:
                # Check if the word appears elsewhere
                if reference_words[i] in hypothesis_words:
                    actual_pos = hypothesis_words.index(reference_words[i])
                    errors.append({
                        "word": reference_words[i],
                        "expected_position": i,
                        "actual_position": actual_pos,
                        "error_type": "order"
                    })
        
        return errors
