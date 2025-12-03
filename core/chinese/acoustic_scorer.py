"""
Task 4: Acoustic Scoring

Scores pronunciation accuracy using WavLM embeddings and cosine similarity.
Evaluates initial/final (声母/韵母) pronunciation quality.
"""

from typing import List, Dict, Optional
import numpy as np
import warnings


class AcousticScorer:
    """
    Acoustic scoring using WavLM-base+ embeddings.
    
    Evaluates pronunciation quality by comparing user audio embeddings
    with reference embeddings using cosine similarity.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize acoustic scorer.
        
        Args:
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.device = device
        self.model = None
        self.processor = None
        self.available = False
        
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch = torch
        except ImportError:
            warnings.warn("PyTorch not available")
            return
        
        # Try to load WavLM model
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            self.FeatureExtractor = Wav2Vec2FeatureExtractor
            self.WavLMModel = WavLMModel
            self.available = True
        except ImportError:
            warnings.warn(
                "Transformers not available. Install with: pip install transformers"
            )
    
    def load_model(self, model_name: str = "microsoft/wavlm-base-plus"):
        """
        Load WavLM model for feature extraction.
        
        Args:
            model_name: HuggingFace model name
        """
        if not self.available:
            raise RuntimeError("Transformers not available")
        
        self.processor = self.FeatureExtractor.from_pretrained(model_name)
        self.model = self.WavLMModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"WavLM model loaded on {self.device}")
    
    def extract_embedding(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract WavLM embedding from audio segment.
        
        Args:
            audio_segment: Audio samples as numpy array
        
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if len(audio_segment) == 0:
            # Return zero embedding for empty segments
            return np.zeros(768)  # WavLM-base+ has 768 dimensions
        
        # Process audio
        inputs = self.processor(
            audio_segment,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with self.torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling over time dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy
        embedding = embeddings.cpu().numpy()[0]
        
        return embedding
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to 0-1 range
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def score_segments(
        self,
        sliced_results: List[Dict],
        reference_audio_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Score acoustic quality of pronunciation segments.
        
        Args:
            sliced_results: Output from AudioSlicer with audio segments
            reference_audio_path: Optional path to reference audio
        
        Returns:
            List with acoustic scores added:
            [
                {"char":"你", "acoustic_score":0.92, ...},
                {"char":"好", "acoustic_score":0.88, ...},
                ...
            ]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        scored_results = []
        
        # If reference audio provided, extract reference embeddings
        reference_embeddings = None
        if reference_audio_path:
            reference_embeddings = self._extract_reference_embeddings(
                reference_audio_path,
                sliced_results
            )
        
        for i, item in enumerate(sliced_results):
            audio_segment = item["audio_segment"]
            
            # Extract embedding
            user_embedding = self.extract_embedding(audio_segment)
            
            # Calculate score
            if reference_embeddings and i < len(reference_embeddings):
                # Compare with reference
                ref_embedding = reference_embeddings[i]
                similarity = self.cosine_similarity(user_embedding, ref_embedding)
                acoustic_score = similarity
            else:
                # No reference: use heuristic scoring
                # Based on embedding magnitude and variance
                acoustic_score = self._heuristic_score(user_embedding)
            
            # Add score to result
            result = item.copy()
            result["acoustic_score"] = float(acoustic_score)
            scored_results.append(result)
        
        return scored_results
    
    def _extract_reference_embeddings(
        self,
        reference_audio_path: str,
        sliced_results: List[Dict]
    ) -> List[np.ndarray]:
        """
        Extract embeddings from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio
            sliced_results: Sliced results with timestamps
        
        Returns:
            List of reference embeddings
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError("librosa not available")
        
        # Load reference audio
        ref_audio, _ = librosa.load(reference_audio_path, sr=16000, mono=True)
        
        ref_embeddings = []
        
        # Extract embeddings for each segment
        for item in sliced_results:
            start_time = item["start"]
            end_time = item["end"]
            
            # Extract segment
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            
            start_sample = max(0, min(start_sample, len(ref_audio)))
            end_sample = max(0, min(end_sample, len(ref_audio)))
            
            if end_sample > start_sample:
                ref_segment = ref_audio[start_sample:end_sample]
            else:
                ref_segment = np.array([])
            
            # Extract embedding
            ref_embedding = self.extract_embedding(ref_segment)
            ref_embeddings.append(ref_embedding)
        
        return ref_embeddings
    
    def _heuristic_score(self, embedding: np.ndarray) -> float:
        """
        Heuristic scoring when no reference is available.
        
        Args:
            embedding: Audio embedding
        
        Returns:
            Heuristic score (0.0 to 1.0)
        """
        # Use embedding statistics as proxy for quality
        # Higher magnitude and moderate variance suggest clear pronunciation
        
        magnitude = np.linalg.norm(embedding)
        variance = np.var(embedding)
        
        # Normalize to 0-1 range (empirical thresholds)
        mag_score = min(1.0, magnitude / 20.0)
        var_score = min(1.0, variance / 0.1)
        
        # Combine scores
        score = 0.7 * mag_score + 0.3 * var_score
        
        return float(score)
    
    def is_available(self) -> bool:
        """Check if WavLM is available."""
        return self.available
