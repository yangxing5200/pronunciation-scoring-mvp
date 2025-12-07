"""
Chinese pronunciation scoring modules.

This package implements a comprehensive 9-task pipeline for Chinese pronunciation scoring:
1. Pinyin mapping (text → phoneme)
2. Audio alignment (forced alignment)
3. Audio slicing
4. Acoustic scoring
5. Tone scoring
6. Duration scoring
7. Pause/fluency scoring
8. Error classification
9. Final comprehensive scoring
"""

from .pinyin_mapper import PinyinMapper
from .audio_aligner_enhanced import ChineseAudioAlignerEnhanced as ChineseAudioAligner
from .audio_slicer import AudioSlicer
from .acoustic_scorer import AcousticScorer
from .tone_scorer import ToneScorer
from .duration_scorer import DurationScorer
from .pause_scorer import PauseScorer
from .error_classifier import ErrorClassifier
from .final_scorer import FinalScorer
from .pipeline import ChineseScoringPipeline

__all__ = [
    'PinyinMapper',
    'ChineseAudioAligner',
    'AudioSlicer',
    'AcousticScorer',
    'ToneScorer',
    'DurationScorer',
    'PauseScorer',
    'ErrorClassifier',
    'FinalScorer',
    'ChineseScoringPipeline',
]
