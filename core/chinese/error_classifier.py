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
        
        self.processor = self.FeatureExtractor.from_pretrained(model_name)
        self.wavlm_model = self.WavLMModel.from_pretrained(model_name)
        self.wavlm_model.to(self.device)
        self.wavlm_model.eval()
        
        self.error_classifier = self._create_error_classifier()
        self.error_classifier.to(self.device)
        self.error_classifier.eval()
        
        print(f"Error classifier loaded on {self.device}")
    
    def _create_error_classifier(self):
        """
        Create MLP for multi-label error classification.
        
        Returns:
            PyTorch MLP model
        """
        nn = self.nn
