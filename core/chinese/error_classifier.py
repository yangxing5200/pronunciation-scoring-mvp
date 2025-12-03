"""
Task 8: Error Classification

Identifies common pronunciation error types for each character.
Uses WavLM embeddings with MLP classifier.
"""

from typing import List, Dict, Optional
import numpy as np
import warnings


class ErrorClassifier:
    """
    Error classification using WavLM embeddings and MLP.
    
    Identifies common pronunciation errors:
    - 声母轻 (initial too soft)
    - 韵母不圆 (final not rounded)
    - 声调错误 (wrong tone)
    - 发音模糊 (unclear pronunciation)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize error classifier.
        
        Args:
            device: Device to use ('cuda' or 'cpu'), auto-detect if None
        """
        self.device = device
        self.wavlm_model = None
        self.processor = None
        self.error_classifier = None
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
        
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            self.FeatureExtractor = Wav2Vec2FeatureExtractor
            self.WavLMModel = WavLMModel
            self.available = True
        except ImportError:
            warnings.warn("Transformers not available")
    
    def load_models(self, model_name: str = "microsoft/wavlm-base-plus"):
        """
        Load WavLM model and initialize error classifier.
        
        Args:
            model_name: HuggingFace model name
        """
        if not self.available:
            raise RuntimeError("Required libraries not available")
        
        # Load WavLM
        self.processor = self.FeatureExtractor.from_pretrained(model_name)
        self.wavlm_model = self.WavLMModel.from_pretrained(model_name)
        self.wavlm_model.to(self.device)
        self.wavlm_model.eval()
        
        # Initialize MLP for multi-label error classification
        self.error_classifier = self._create_error_classifier()
        self.error_classifier.to(self.device)
        
        print(f"Error classifier loaded on {self.device}")
    
    def _create_error_classifier(self):
        """
        Create MLP for multi-label error classification.
        
        Returns:
            PyTorch MLP model
        """
        class ErrorMLP(self.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = self.nn.Linear(768, 256)
                self.relu1 = self.nn.ReLU()
                self.dropout1 = self.nn.Dropout(0.3)
                self.fc2 = self.nn.Linear(256, 64)
                self.relu2 = self.nn.ReLU()
                self.dropout2 = self.nn.Dropout(0.3)
                self.fc3 = self.nn.Linear(64, 4)  # 4 error types
                self.sigmoid = self.nn.Sigmoid()  # Multi-label
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                x = self.fc3(x)
                x = self.sigmoid(x)
                return x
        
        return ErrorMLP()
    
    def extract_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract features using WavLM.
        
        Args:
            audio_segment: Audio samples
        
        Returns:
            Feature vector
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
        
        # Extract features
        with self.torch.no_grad():
            outputs = self.wavlm_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1)
        
        return features.cpu().numpy()[0]
    
    def classify_errors(
        self,
        sliced_results: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        Classify pronunciation errors for each character.
        
        Args:
            sliced_results: Output from AudioSlicer
            threshold: Confidence threshold for error detection
        
        Returns:
            List with error classifications added:
            [
                {"char":"你", "errors":["声母轻", "韵母不圆"], ...},
                {"char":"好", "errors":[], ...},
                ...
            ]
        """
        if self.wavlm_model is None or self.error_classifier is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Error type labels
        error_labels = [
            "声母轻",      # Initial too soft
            "韵母不圆",    # Final not rounded
            "声调错误",    # Wrong tone
            "发音模糊"     # Unclear pronunciation
        ]
        
        classified_results = []
        
        for item in sliced_results:
            audio_segment = item["audio_segment"]
            
            # Extract features
            features = self.extract_features(audio_segment)
            
            # Classify errors
            features_tensor = self.torch.tensor(
                features,
                dtype=self.torch.float32
            ).unsqueeze(0).to(self.device)
            
            with self.torch.no_grad():
                predictions = self.error_classifier(features_tensor)
                predictions = predictions.cpu().numpy()[0]
            
            # Extract errors above threshold
            errors = []
            for i, pred in enumerate(predictions):
                if pred > threshold:
                    errors.append(error_labels[i])
            
            # Use heuristic rules to enhance classification
            errors = self._apply_heuristic_rules(item, errors)
            
            # Add to result
            result = item.copy()
            result["errors"] = errors
            result["error_probabilities"] = {
                error_labels[i]: float(predictions[i])
                for i in range(len(error_labels))
            }
            classified_results.append(result)
        
        return classified_results
    
    def _apply_heuristic_rules(
        self,
        item: Dict,
        ml_errors: List[str]
    ) -> List[str]:
        """
        Apply heuristic rules to refine error classification.
        
        Args:
            item: Item with scores and features
            ml_errors: Errors from ML classifier
        
        Returns:
            Refined error list
        """
        errors = ml_errors.copy()
        
        # Check tone score if available
        if "tone_score" in item and item["tone_score"] < 0.6:
            if "声调错误" not in errors:
                errors.append("声调错误")
        
        # Check acoustic score if available
        if "acoustic_score" in item and item["acoustic_score"] < 0.5:
            if "发音模糊" not in errors:
                errors.append("发音模糊")
        
        # Check duration - very short might indicate unclear pronunciation
        if "duration" in item and item["duration"] < 0.1:
            if "发音模糊" not in errors:
                errors.append("发音模糊")
        
        return errors
    
    def is_available(self) -> bool:
        """Check if error classifier is available."""
        return self.available
