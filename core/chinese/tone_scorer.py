"""
Task 5: Tone Scoring

Scores tone accuracy for each Chinese character.
Uses WavLM prosody embeddings with MLP classifier.
"""

from typing import List, Dict, Optional
import numpy as np
import warnings


class ToneScorer:
    """
    Tone scoring using WavLM prosody features and MLP classifier.
    
    Evaluates whether each character's tone (1-4, or neutral) is correct.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize tone scorer.
        
        Args:
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.device = device
        self.wavlm_model = None
        self.processor = None
        self.tone_classifier = None
        self.available = False
        
        try:
            import torch
            import torch.nn as nn
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.torch = torch
            self.nn = nn
        except ImportError:
            warnings.warn("PyTorch not available")
            return
        
        # Try to load models
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            self.FeatureExtractor = Wav2Vec2FeatureExtractor
            self.WavLMModel = WavLMModel
            self.available = True
        except ImportError:
            warnings.warn("Transformers not available")
    
    def load_models(self, model_name: str = "microsoft/wavlm-base-plus"):
        """
        Load WavLM model and initialize tone classifier.
        
        Args:
            model_name: HuggingFace model name
        """
        if not self.available:
            raise RuntimeError("Required libraries not available")
        
        # Load WavLM for prosody features
        self.processor = self.FeatureExtractor.from_pretrained(model_name)
        self.wavlm_model = self.WavLMModel.from_pretrained(model_name)
        self.wavlm_model.to(self.device)
        self.wavlm_model.eval()
        
        # Initialize simple MLP classifier for tone classification
        # 5 classes: tone 1, 2, 3, 4, and neutral (5)
        self.tone_classifier = self._create_tone_classifier()
        self.tone_classifier.to(self.device)
        
        print(f"Tone scoring models loaded on {self.device}")
    
    def _create_tone_classifier(self):
        """
        Create a simple MLP for tone classification.
        
        Returns:
            PyTorch MLP model
        """
        class ToneMLP(self.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = self.nn.Linear(768, 256)  # WavLM-base+ is 768-dim
                self.relu1 = self.nn.ReLU()
                self.dropout1 = self.nn.Dropout(0.3)
                self.fc2 = self.nn.Linear(256, 64)
                self.relu2 = self.nn.ReLU()
                self.dropout2 = self.nn.Dropout(0.3)
                self.fc3 = self.nn.Linear(64, 5)  # 5 tone classes
                self.softmax = self.nn.Softmax(dim=1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x = self.fc3(x)
                x = self.softmax(x)
                return x
        
        return ToneMLP()
    
    def extract_prosody_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract prosody features using WavLM.
        
        Args:
            audio_segment: Audio samples as numpy array
        
        Returns:
            Prosody feature vector
        """
        if self.wavlm_model is None:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        
        if len(audio_segment) == 0:
            return np.zeros(768)
        
        # Process audio
        inputs = self.processor(
            audio_segment,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract WavLM features
        with self.torch.no_grad():
            outputs = self.wavlm_model(**inputs)
            # Mean pooling over time
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.cpu().numpy()[0]
    
    def predict_tone(self, prosody_features: np.ndarray) -> int:
        """
        Predict tone from prosody features.
        
        Args:
            prosody_features: Prosody feature vector
        
        Returns:
            Predicted tone number (1-5)
        """
        if self.tone_classifier is None:
            raise RuntimeError("Classifier not loaded. Call load_models() first.")
        
        # Convert to tensor
        features_tensor = self.torch.tensor(
            prosody_features,
            dtype=self.torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Predict
        with self.torch.no_grad():
            output = self.tone_classifier(features_tensor)
            predicted_class = self.torch.argmax(output, dim=1).item()
        
        # Convert class index to tone number (0->1, 1->2, 2->3, 3->4, 4->5)
        return predicted_class + 1
    
    def score_tones(
        self,
        sliced_results: List[Dict]
    ) -> List[Dict]:
        """
        Score tone accuracy for each character.
        
        Args:
            sliced_results: Output from AudioSlicer with pinyin info
        
        Returns:
            List with tone scores added:
            [
                {"char":"你", "tone_score":0.95, "predicted_tone":3, ...},
                {"char":"好", "tone_score":0.90, "predicted_tone":3, ...},
                ...
            ]
        """
        if self.wavlm_model is None or self.tone_classifier is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        scored_results = []
        
        for item in sliced_results:
            audio_segment = item["audio_segment"]
            pinyin = item["pinyin"]
            
            # Extract expected tone from pinyin
            expected_tone = self._extract_tone_from_pinyin(pinyin)
            
            # Extract prosody features
            prosody_features = self.extract_prosody_features(audio_segment)
            
            # Predict tone
            predicted_tone = self.predict_tone(prosody_features)
            
            # Calculate tone score
            if expected_tone == predicted_tone:
                tone_score = 1.0  # Perfect match
            elif abs(expected_tone - predicted_tone) == 1:
                tone_score = 0.7  # Close tone
            else:
                tone_score = 0.4  # Incorrect tone
            
            # If audio segment is too short, reduce confidence
            if len(audio_segment) < 1600:  # Less than 0.1 seconds
                tone_score *= 0.5
            
            # Add to result
            result = item.copy()
            result["tone_score"] = float(tone_score)
            result["predicted_tone"] = predicted_tone
            result["expected_tone"] = expected_tone
            scored_results.append(result)
        
        return scored_results
    
    def _extract_tone_from_pinyin(self, pinyin: str) -> int:
        """
        Extract tone number from pinyin string.
        
        Args:
            pinyin: Pinyin with tone number (e.g., "ni3", "hao3")
        
        Returns:
            Tone number (1-5)
        """
        import re
        match = re.search(r'(\d)$', pinyin)
        if match:
            return int(match.group(1))
        return 5  # Default to neutral tone
    
    def is_available(self) -> bool:
        """Check if tone scorer is available."""
        return self.available
