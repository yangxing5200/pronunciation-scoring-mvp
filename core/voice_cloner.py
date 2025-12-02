"""
Voice cloning module using IndexTTS2 for timbre transfer. 
Uses subprocess to call index-tts CLI to avoid dependency conflicts.
"""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union


class VoiceCloner:
    """
    Voice cloner using IndexTTS2 for timbre transfer. 
    Uses subprocess to call index-tts in its own environment.
    """
    
    def __init__(
        self,
        model_dir: str = "models/indextts2",
        device: str = "cuda",
        use_fp16: bool = False,
        index_tts_path: str = "D:/workspace/dev/github/index-tts"  # 你的 index-tts 路径
    ):
        """
        Initialize voice cloner.
        
        Args:
            model_dir: Directory containing IndexTTS2 model files (unused, kept for compatibility)
            device: Device to use (cpu/cuda)
            use_fp16: Whether to use FP16 precision
            index_tts_path: Path to index-tts project directory
        """
        self.index_tts_path = Path(index_tts_path)
        self.device = device
        self.use_fp16 = use_fp16
        self.model_available = False
        
        self._check_availability()
    
    def _check_availability(self):
        """Check if index-tts is available."""
        checkpoints_path = self.index_tts_path / "checkpoints"
        config_path = checkpoints_path / "config.yaml"
        
        if not self.index_tts_path.exists():
            warnings.warn(f"index-tts directory not found at {self.index_tts_path}")
            return
        
        if not config_path.exists():
            warnings. warn(f"index-tts config not found at {config_path}")
            return
        
        self.model_available = True
        print(f"✓ IndexTTS2 available at {self.index_tts_path}")
    
    def clone_voice(
        self,
        text: str,
        reference_audio_path: Union[str, Path],
        output_path: Union[str, Path],
        language: str = "en"
    ) -> bool:
        """
        Generate speech with cloned voice timbre.
        
        Args:
            text: Text to synthesize
            reference_audio_path: Path to reference audio (user's voice)
            output_path: Path to save generated audio
            language: Target language
        
        Returns:
            True if successful, False otherwise
        """
        reference_audio_path = Path(reference_audio_path). resolve()
        output_path = Path(output_path).resolve()
        
        if not reference_audio_path. exists():
            raise FileNotFoundError(f"Reference audio not found: {reference_audio_path}")
        
        # Ensure output directory exists
        output_path. parent.mkdir(parents=True, exist_ok=True)
        
        if not self.model_available:
            warnings.warn("Voice cloning not available.  IndexTTS2 not found.")
            return False
        
        try:
            print(f"IndexTTS2: Generating speech for '{text}' using reference {reference_audio_path. name}")
            
            # 构建 Python 脚本
            python_script = f'''
import sys
sys.path.insert(0, "{self.index_tts_path. as_posix()}")
from indextts. infer_v2 import IndexTTS2

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
            result = subprocess. run(
                ["uv", "run", "python", "-c", python_script],
                cwd=str(self.index_tts_path),
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
            
            if result.returncode != 0:
                print(f"IndexTTS2 stderr: {result.stderr}")
                warnings.warn(f"IndexTTS2 inference failed: {result. stderr}")
                return False
            
            # Verify output was created
            if output_path.exists() and output_path. stat().st_size > 0:
                print(f"IndexTTS2: Successfully generated audio at {output_path}")
                return True
            else:
                warnings.warn("IndexTTS2: Output file not created or empty")
                return False
                
        except subprocess.TimeoutExpired:
            warnings.warn("IndexTTS2 inference timed out")
            return False
        except Exception as e:
            warnings. warn(f"IndexTTS2 inference failed: {e}")
            return False
    
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