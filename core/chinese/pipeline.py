"""
Chinese Pronunciation Scoring Pipeline

Integrates all 9 tasks into a complete pronunciation scoring system.
"""

from typing import Dict, List, Optional
from pathlib import Path
import warnings

from .pinyin_mapper import PinyinMapper
from .audio_aligner import ChineseAudioAligner
from .audio_slicer import AudioSlicer
from .acoustic_scorer import AcousticScorer
from .tone_scorer import ToneScorer
from .duration_scorer import DurationScorer
from .pause_scorer import PauseScorer
from .error_classifier import ErrorClassifier
from .final_scorer import FinalScorer


class ChineseScoringPipeline:
    """
    Complete Chinese pronunciation scoring pipeline.
    
    Executes all 9 tasks in sequence:
    1. Pinyin mapping
    2. Audio alignment
    3. Audio slicing
    4. Acoustic scoring
    5. Tone scoring
    6. Duration scoring
    7. Pause scoring
    8. Error classification
    9. Final scoring
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        enable_gpu: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
            enable_gpu: Whether to enable GPU acceleration
        """
        # Determine device
        if device is None and enable_gpu:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        elif device is None:
            device = "cpu"
        
        self.device = device
        
        # Initialize all modules
        self.pinyin_mapper = PinyinMapper()
        self.audio_aligner = ChineseAudioAligner(device=device)
        self.audio_slicer = AudioSlicer()
        self.acoustic_scorer = AcousticScorer(device=device)
        self.tone_scorer = ToneScorer(device=device)
        self.duration_scorer = DurationScorer()
        self.pause_scorer = PauseScorer()
        self.error_classifier = ErrorClassifier(device=device)
        self.final_scorer = FinalScorer()
        
        self.models_loaded = False
    
    def load_models(self, model_size: str = "base"):
        """
        Load all required models.
        
        Args:
            model_size: Whisper model size for alignment
        """
        if self.models_loaded:
            return
        
        print("Loading Chinese pronunciation scoring models...")
        
        # Load alignment models
        if self.audio_aligner.is_available():
            try:
                self.audio_aligner.load_models(model_size)
            except Exception as e:
                warnings.warn(f"Failed to load alignment models: {e}")
        
        # Load acoustic scoring models
        if self.acoustic_scorer.is_available():
            try:
                self.acoustic_scorer.load_model()
            except Exception as e:
                warnings.warn(f"Failed to load acoustic models: {e}")
        
        # Load tone scoring models
        if self.tone_scorer.is_available():
            try:
                self.tone_scorer.load_models()
            except Exception as e:
                warnings.warn(f"Failed to load tone models: {e}")
        
        # Load error classification models
        if self.error_classifier.is_available():
            try:
                self.error_classifier.load_models()
            except Exception as e:
                warnings.warn(f"Failed to load error classifier: {e}")
        
        self.models_loaded = True
        print(f"Models loaded on {self.device}")
    
    def score_pronunciation(
        self,
        audio_path: str,
        reference_text: str,
        reference_audio_path: Optional[str] = None
    ) -> Dict:
        """
        Score Chinese pronunciation using complete pipeline.
        
        Args:
            audio_path: Path to user's audio file
            reference_text: Expected Chinese text
            reference_audio_path: Optional path to reference audio
        
        Returns:
            Complete scoring results dictionary
        """
        if not self.models_loaded:
            self.load_models()
        
        # Task 1: Pinyin Mapping
        print("Task 1: Mapping text to pinyin...")
        pinyin_sequence = self.pinyin_mapper.text_to_pinyin(reference_text)
        
        if not pinyin_sequence:
            raise ValueError("No Chinese characters found in reference text")
        
        # Task 2: Audio Alignment
        print("Task 2: Aligning audio with text...")
        if self.audio_aligner.is_available():
            alignment_results = self.audio_aligner.align_audio(
                audio_path,
                pinyin_sequence
            )
        else:
            # Fallback: create placeholder alignment
            warnings.warn("WhisperX not available, using placeholder alignment")
            alignment_results = self._create_placeholder_alignment(
                audio_path,
                pinyin_sequence
            )
        
        # Task 3: Audio Slicing
        print("Task 3: Slicing audio into character segments...")
        sliced_results = self.audio_slicer.slice_audio(
            audio_path,
            alignment_results
        )
        
        # Task 4: Acoustic Scoring
        print("Task 4: Scoring acoustic quality...")
        if self.acoustic_scorer.is_available():
            sliced_results = self.acoustic_scorer.score_segments(
                sliced_results,
                reference_audio_path
            )
        else:
            # Add placeholder scores
            for item in sliced_results:
                item["acoustic_score"] = 0.7
        
        # Task 5: Tone Scoring
        print("Task 5: Scoring tone accuracy...")
        if self.tone_scorer.is_available():
            sliced_results = self.tone_scorer.score_tones(sliced_results)
        else:
            # Add placeholder scores
            for item in sliced_results:
                item["tone_score"] = 0.7
                item["predicted_tone"] = 0
                item["expected_tone"] = 0
        
        # Task 6: Duration Scoring
        print("Task 6: Scoring pronunciation duration...")
        sliced_results = self.duration_scorer.score_durations(sliced_results)
        
        # Task 7: Pause/Fluency Scoring
        print("Task 7: Scoring fluency and pauses...")
        sliced_results = self.pause_scorer.score_pauses(sliced_results)
        fluency_metrics = self.pause_scorer.calculate_overall_fluency(sliced_results)
        
        # Task 8: Error Classification
        print("Task 8: Classifying pronunciation errors...")
        if self.error_classifier.is_available():
            sliced_results = self.error_classifier.classify_errors(sliced_results)
        else:
            # Add placeholder errors
            for item in sliced_results:
                item["errors"] = []
                item["error_probabilities"] = {}
        
        # Task 9: Final Scoring
        print("Task 9: Calculating final scores...")
        final_results = self.final_scorer.calculate_final_scores(sliced_results)
        overall_metrics = self.final_scorer.calculate_overall_score(final_results)
        feedback = self.final_scorer.generate_feedback(final_results, overall_metrics)
        
        # Compile complete results
        results = {
            "overall_score": overall_metrics["overall_score"],
            "character_scores": final_results,
            "overall_metrics": overall_metrics,
            "fluency_metrics": fluency_metrics,
            "feedback": feedback,
            "reference_text": reference_text,
            "num_characters": len(final_results)
        }
        
        print(f"Scoring complete! Overall score: {results['overall_score']}/100")
        
        return results
    
    def _create_placeholder_alignment(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict]
    ) -> List[Dict]:
        """
        Create placeholder alignment when WhisperX is not available.
        
        Args:
            audio_path: Path to audio file
            pinyin_sequence: Pinyin sequence from Task 1
        
        Returns:
            Placeholder alignment results
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError("librosa required for placeholder alignment")
        
        # Load audio to get duration
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(audio) / 16000
        
        # Divide duration equally among characters
        num_chars = len(pinyin_sequence)
        char_duration = total_duration / num_chars if num_chars > 0 else 0
        
        alignment_results = []
        for i, item in enumerate(pinyin_sequence):
            alignment_results.append({
                "char": item["char"],
                "pinyin": item["pinyin"],
                "start": i * char_duration,
                "end": (i + 1) * char_duration,
                "score": 0.5  # Placeholder confidence
            })
        
        return alignment_results
    
    def is_available(self) -> bool:
        """
        Check if pipeline is available (pypinyin installed).
        
        Returns:
            True if pipeline can run
        """
        return self.pinyin_mapper.is_available()
