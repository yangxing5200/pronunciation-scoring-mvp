"""
Voice cloning module using IndexTTS2 for timbre transfer. 
Uses HTTP API for fast inference, with subprocess fallback.
"""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union
import requests


class VoiceCloner:
    """
    Voice cloner using IndexTTS2 for timbre transfer. 
    Uses HTTP API for fast inference, with subprocess fallback.
    """
    
    def __init__(
        self,
        model_dir: str = "models/indextts2",
        device: str = "cuda",
        use_fp16: bool = False,
        index_tts_path: str = "D:/workspace/dev/github/index-tts",  # 你的 index-tts 路径
        api_url: str = "http://localhost:5000"
    ):
        """
        Initialize voice cloner.
        
        Args:
            model_dir: Directory containing IndexTTS2 model files (unused, kept for compatibility)
            device: Device to use (cpu/cuda)
            use_fp16: Whether to use FP16 precision
            index_tts_path: Path to index-tts project directory
            api_url: URL of the IndexTTS2 HTTP API server
        """
        self.index_tts_path = Path(index_tts_path)
        self.device = device
        self.use_fp16 = use_fp16
        self.api_url = api_url
        self.model_available = False
        self.api_available = False
        
        self._check_availability()
    
    def _check_availability(self):
        """Check if index-tts API or subprocess is available."""
        # Check API availability
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=2
            )
            if response.status_code == 200:
                self.api_available = True
                print(f"✓ IndexTTS2 HTTP API available at {self.api_url}")
                self.model_available = True
                return
        except (requests.RequestException, Exception):
            self.api_available = False
        
        # Fallback: Check subprocess availability
        checkpoints_path = self.index_tts_path / "checkpoints"
        config_path = checkpoints_path / "config.yaml"
        
        if not self.index_tts_path.exists():
            warnings.warn(f"index-tts directory not found at {self.index_tts_path}")
            return
        
        if not config_path.exists():
            warnings.warn(f"index-tts config not found at {config_path}")
            return
        
        self.model_available = True
        print(f"✓ IndexTTS2 subprocess available at {self.index_tts_path}")
        print(f"⚠ HTTP API not available, will use slower subprocess method")
    
    def _clone_voice_via_api(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path
    ) -> bool:
        """
        Generate speech using HTTP API.
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio
            output_path: Path to save generated audio
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"IndexTTS2 API: Generating speech for '{text}'")
            
            # Prepare multipart form data
            with open(reference_audio_path, 'rb') as audio_file:
                files = {
                    'reference_audio': (reference_audio_path.name, audio_file, 'audio/wav')
                }
                data = {
                    'text': text
                }
                
                # POST request to TTS endpoint with 120 second timeout
                response = requests.post(
                    f"{self.api_url}/tts",
                    data=data,
                    files=files,
                    timeout=120
                )
            
            if response.status_code == 200:
                # Save the audio response
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                # Verify output was created
                if output_path.exists() and output_path.stat().st_size > 0:
                    print(f"IndexTTS2 API: Successfully generated audio at {output_path}")
                    return True
                else:
                    warnings.warn("IndexTTS2 API: Output file not created or empty")
                    return False
            else:
                warnings.warn(f"IndexTTS2 API request failed with status {response.status_code}")
                return False
                
        except requests.Timeout:
            warnings.warn("IndexTTS2 API request timed out")
            return False
        except Exception as e:
            warnings.warn(f"IndexTTS2 API request failed: {e}")
            return False
    
    def _clone_voice_via_subprocess(
        self,
        text: str,
        reference_audio_path: Path,
        output_path: Path
    ) -> bool:
        """
        Generate speech using subprocess (fallback method).
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio
            output_path: Path to save generated audio
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"IndexTTS2 subprocess: Generating speech for '{text}'")
            
            # 构建 Python 脚本
            python_script = f'''
import sys
sys.path.insert(0, "{self.index_tts_path.as_posix()}")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16={self.use_fp16}
)
tts.infer(
    spk_audio_prompt=r"{reference_audio_path}",
    text=r"{text}",
    output_path=r"{output_path}",
    verbose=True
)
print("SUCCESS")
'''
            
            # 使用 uv run 在 index-tts 环境中执行
            result = subprocess.run(
                ["uv", "run", "python", "-c", python_script],
                cwd=str(self.index_tts_path),
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
            
            if result.returncode != 0:
                print(f"IndexTTS2 stderr: {result.stderr}")
                warnings.warn(f"IndexTTS2 inference failed: {result.stderr}")
                return False
            
            # Verify output was created
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"IndexTTS2 subprocess: Successfully generated audio at {output_path}")
                return True
            else:
                warnings.warn("IndexTTS2 subprocess: Output file not created or empty")
                return False
                
        except subprocess.TimeoutExpired:
            warnings.warn("IndexTTS2 subprocess timed out")
            return False
        except Exception as e:
            warnings.warn(f"IndexTTS2 subprocess failed: {e}")
            return False
    
    def clone_voice(
        self,
        text: str,
        reference_audio_path: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "en"
    ) -> bool:
        """
        Generate speech with cloned voice timbre.
        Uses HTTP API if available, falls back to subprocess.
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio (user's voice)
            output_path: Path to save generated audio
            language: Target language
        
        Returns:
            True if successful, False otherwise
        """
        reference_audio_path = Path(reference_audio_path).resolve()
        output_path = Path(output_path).resolve()
        
        if not reference_audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.model_available:
            warnings.warn("Voice cloning not available. IndexTTS2 not found.")
            return False
        
        # Try API first if available
        if self.api_available:
            success = self._clone_voice_via_api(text, reference_audio_path, output_path)
            if success:
                return True
            
            # If API fails, try subprocess fallback
            print("⚠ API method failed, trying subprocess fallback...")
        
        # Use subprocess method
        return self._clone_voice_via_subprocess(text, reference_audio_path, output_path)
    
    def generate_standard_pronunciation(
        self,
        text: str,
        user_audio_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Generate standard pronunciation in user's voice.
        """
        return self.clone_voice(
            text=text,
            reference_audio_path=user_audio_path,
            output_path=output_path
        )
    
    def is_available(self) -> bool:
        """
        Check if voice cloning is available. 
        """
        return self.model_available