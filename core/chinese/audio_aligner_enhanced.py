"""
Task 2: Audio Alignment - Enhanced Version with Boundary Refinement

增强版音频对齐器：
- 使用 FunASR 获取初始时间戳
- 基于能量和过零率优化字符边界
- 减少相邻字符的声音混叠
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
import numpy as np
import re


class BoundaryRefiner:
    """
    基于声学特征的边界优化器。
    
    在 ASR 返回的粗略时间戳基础上，使用能量和过零率来精细调整边界。
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.frame_size = 256  # 16ms @ 16kHz
        self.hop_size = 64     # 4ms @ 16kHz
        
        # 边界搜索范围（秒）
        self.search_range = 0.05  # 50ms
        
        # 最小间隔（秒）- 避免相邻字符重叠
        self.min_gap = 0.02  # 20ms
    
    def compute_energy(self, audio: np.ndarray) -> np.ndarray:
        """计算短时能量。"""
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        energy = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            energy[i] = np.sum(frame ** 2) / self.frame_size
        
        return energy
    
    def compute_zcr(self, audio: np.ndarray) -> np.ndarray:
        """计算过零率。"""
        n_frames = (len(audio) - self.frame_size) // self.hop_size + 1
        zcr = np.zeros(n_frames)
        
        for i in range(n_frames):
            start = i * self.hop_size
            frame = audio[start:start + self.frame_size]
            # 计算符号变化次数
            signs = np.sign(frame)
            zcr[i] = np.sum(np.abs(np.diff(signs))) / (2 * self.frame_size)
        
        return zcr
    
    def compute_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """计算频谱变化率（用于检测音素边界）。"""
        try:
            import librosa
            # 计算短时傅里叶变换
            stft = librosa.stft(audio, n_fft=512, hop_length=self.hop_size)
            magnitude = np.abs(stft)
            
            # 计算帧间差异
            flux = np.zeros(magnitude.shape[1])
            for i in range(1, magnitude.shape[1]):
                diff = magnitude[:, i] - magnitude[:, i-1]
                flux[i] = np.sum(np.maximum(diff, 0))  # 只考虑正向变化
            
            return flux
        except:
            # librosa 不可用，返回空数组
            return np.array([])
    
    def find_boundary(
        self,
        audio: np.ndarray,
        initial_time: float,
        is_start: bool,
        prev_end: Optional[float] = None,
        next_start: Optional[float] = None
    ) -> float:
        """
        寻找最佳边界点。
        
        Args:
            audio: 完整音频
            initial_time: ASR 返回的初始时间点
            is_start: 是否是起始边界（否则是结束边界）
            prev_end: 上一个字符的结束时间
            next_start: 下一个字符的开始时间
        
        Returns:
            优化后的时间点
        """
        initial_sample = int(initial_time * self.sample_rate)
        search_samples = int(self.search_range * self.sample_rate)
        
        # 确定搜索范围
        if is_start:
            # 起始边界：向前搜索
            search_start = max(0, initial_sample - search_samples)
            search_end = min(len(audio), initial_sample + search_samples // 2)
            
            # 不能早于上一个字符的结束
            if prev_end is not None:
                min_sample = int((prev_end + self.min_gap) * self.sample_rate)
                search_start = max(search_start, min_sample)
        else:
            # 结束边界：向后搜索
            search_start = max(0, initial_sample - search_samples // 2)
            search_end = min(len(audio), initial_sample + search_samples)
            
            # 不能晚于下一个字符的开始
            if next_start is not None:
                max_sample = int((next_start - self.min_gap) * self.sample_rate)
                search_end = min(search_end, max_sample)
        
        if search_end <= search_start:
            return initial_time
        
        # 提取搜索区域的特征
        segment = audio[search_start:search_end]
        if len(segment) < self.frame_size:
            return initial_time
        
        energy = self.compute_energy(segment)
        zcr = self.compute_zcr(segment)
        
        if len(energy) == 0:
            return initial_time
        
        # 归一化
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-8)
        
        # 计算边界分数
        # 对于起始边界：寻找能量从低到高的跳变点
        # 对于结束边界：寻找能量从高到低的跳变点
        if is_start:
            # 起始边界：低能量 + 能量上升
            boundary_score = (1 - energy_norm[:-1]) * 0.5 + np.diff(energy_norm) * 0.5
        else:
            # 结束边界：能量下降
            boundary_score = -np.diff(energy_norm) * 0.7 + (1 - energy_norm[:-1]) * 0.3
        
        if len(boundary_score) == 0:
            return initial_time
        
        # 找到最佳边界点
        best_frame = np.argmax(boundary_score)
        best_sample = search_start + best_frame * self.hop_size
        best_time = best_sample / self.sample_rate
        
        # 确保边界在合理范围内
        if is_start and prev_end is not None:
            best_time = max(best_time, prev_end + self.min_gap)
        if not is_start and next_start is not None:
            best_time = min(best_time, next_start - self.min_gap)
        
        return best_time
    
    def refine_boundaries(
        self,
        audio: np.ndarray,
        timestamps: List[Dict]
    ) -> List[Dict]:
        """
        优化所有字符的边界。
        
        Args:
            audio: 完整音频
            timestamps: 初始时间戳列表 [{'char': '你', 'start': 0.1, 'end': 0.3}, ...]
        
        Returns:
            优化后的时间戳列表
        """
        if len(timestamps) == 0:
            return timestamps
        
        refined = []
        
        for i, ts in enumerate(timestamps):
            # 获取相邻字符的时间信息
            prev_end = refined[-1]['end'] if i > 0 else None
            next_start = timestamps[i + 1]['start'] if i < len(timestamps) - 1 else None
            
            # 优化起始边界
            new_start = self.find_boundary(
                audio,
                ts['start'],
                is_start=True,
                prev_end=prev_end
            )
            
            # 优化结束边界
            new_end = self.find_boundary(
                audio,
                ts['end'],
                is_start=False,
                next_start=next_start
            )
            
            # 确保 start < end
            if new_end <= new_start:
                new_end = new_start + 0.05  # 最小 50ms
            
            refined.append({
                **ts,
                'start': new_start,
                'end': new_end,
                'original_start': ts['start'],
                'original_end': ts['end']
            })
        
        return refined
    
    def remove_overlap(
        self,
        timestamps: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """
        移除相邻字符的时间重叠。
        
        Args:
            timestamps: 时间戳列表
            audio_duration: 音频总时长
        
        Returns:
            无重叠的时间戳列表
        """
        if len(timestamps) <= 1:
            return timestamps
        
        result = []
        
        for i, ts in enumerate(timestamps):
            start = ts['start']
            end = ts['end']
            
            # 检查与上一个字符的重叠
            if i > 0 and start < result[-1]['end']:
                # 取中点作为分界
                mid = (result[-1]['start'] + end) / 2
                # 确保每个字符至少有 30ms
                min_duration = 0.03
                
                if mid - result[-1]['start'] >= min_duration and end - mid >= min_duration:
                    result[-1]['end'] = mid - 0.005  # 5ms 间隔
                    start = mid + 0.005
                else:
                    # 无法分割，保持原样
                    start = result[-1]['end'] + 0.01
            
            # 确保不超出音频边界
            end = min(end, audio_duration)
            start = min(start, audio_duration - 0.01)
            
            result.append({
                **ts,
                'start': start,
                'end': end
            })
        
        return result


class ChineseAudioAlignerEnhanced:
    """
    增强版中文音频对齐器。
    
    在 FunASR 对齐基础上添加边界优化。
    """
    
    # 字符段最小/最大时长（秒）
    MIN_CHAR_DURATION = 0.05  # 50ms
    MAX_CHAR_DURATION = 1.0   # 1s
    DEFAULT_CHAR_DURATION = 0.25  # 250ms
    
    def __init__(self, device: Optional[str] = None):
        """初始化对齐器。"""
        self.device = device
        self.funasr_available = False
        self.funasr_model = None
        self.boundary_refiner = BoundaryRefiner()
        
        # 自动检测设备
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ 使用设备: {self.device}")
        except ImportError:
            self.device = "cpu"
        
        # 检测 FunASR
        try:
            from funasr import AutoModel
            self.AutoModel = AutoModel
            self.funasr_available = True
            print("✅ FunASR 可用")
        except ImportError:
            warnings.warn("FunASR 不可用。安装命令: pip install funasr modelscope")
    
    def load_models(self, model_size: str = "base"):
        """加载 ASR 模型。"""
        if self.funasr_available:
            try:
                print("📥 加载 FunASR Paraformer 模型...")
                self.funasr_model = self.AutoModel(
                    model="paraformer-zh",
                    model_revision="v2.0.4",
                    vad_model="fsmn-vad",
                    vad_model_revision="v2.0.4",
                    punc_model="ct-punc",
                    punc_model_revision="v2.0.4",
                    device=self.device
                )
                print(f"✅ FunASR 模型加载完成")
            except Exception as e:
                warnings.warn(f"FunASR 加载失败: {e}")
                self.funasr_available = False
    
    def align_audio(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict[str, str]],
        refine_boundaries: bool = True
    ) -> List[Dict]:
        """
        对齐音频与拼音序列。
        
        Args:
            audio_path: 音频文件路径
            pinyin_sequence: 拼音序列
            refine_boundaries: 是否优化边界
        
        Returns:
            对齐结果列表
        """
        # 加载音频
        audio, audio_duration = self._load_audio(audio_path)
        
        if self.funasr_model is None:
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        # FunASR 识别
        try:
            result = self.funasr_model.generate(
                input=audio_path,
                batch_size_s=300,
                return_raw_text=False,
            )
        except Exception as e:
            warnings.warn(f"FunASR 识别失败: {e}")
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        if not result:
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        # 解析结果
        funasr_result = result[0] if isinstance(result, list) else result
        recognized_text = funasr_result.get('text', '')
        timestamps = funasr_result.get('timestamp', [])
        
        print(f"🎤 识别结果: {recognized_text}")
        print(f"   期望文本: {''.join([item['char'] for item in pinyin_sequence])}")
        
        # 构建初始时间戳
        char_timestamps = self._build_char_timestamps(
            recognized_text, timestamps, audio_duration
        )
        
        # 与期望序列对齐
        aligned_results = self._align_with_expected_sequence(
            char_timestamps, pinyin_sequence, audio_duration
        )
        
        # 边界优化
        if refine_boundaries and len(audio) > 0:
            print("🔧 正在优化字符边界...")
            aligned_results = self.boundary_refiner.refine_boundaries(
                audio, aligned_results
            )
            aligned_results = self.boundary_refiner.remove_overlap(
                aligned_results, audio_duration
            )
            print("✅ 边界优化完成")
        
        # 后处理
        aligned_results = self._postprocess_timestamps(aligned_results, audio_duration)
        
        return aligned_results
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """加载音频文件。"""
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = len(audio) / sr
            return audio, duration
        except Exception as e:
            warnings.warn(f"加载音频失败: {e}")
            return np.array([]), 5.0
    
    def _build_char_timestamps(
        self,
        recognized_text: str,
        timestamps: List,
        audio_duration: float
    ) -> List[Dict]:
        """构建字符时间戳。"""
        char_timestamps = []
        
        if timestamps and len(timestamps) == len(recognized_text):
            for i, char in enumerate(recognized_text):
                if re.match(r'[\u4e00-\u9fff]', char):
                    ts = timestamps[i]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start = ts[0] / 1000.0
                        end = ts[1] / 1000.0
                        char_timestamps.append({
                            'char': char,
                            'start': start,
                            'end': end,
                            'score': 0.9
                        })
        elif timestamps:
            # 时间戳数量不匹配
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', recognized_text)
            if len(timestamps) == len(chinese_chars):
                for i, char in enumerate(chinese_chars):
                    ts = timestamps[i]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        char_timestamps.append({
                            'char': char,
                            'start': ts[0] / 1000.0,
                            'end': ts[1] / 1000.0,
                            'score': 0.85
                        })
            else:
                # 均匀分配
                char_duration = audio_duration / max(1, len(chinese_chars))
                for i, char in enumerate(chinese_chars):
                    char_timestamps.append({
                        'char': char,
                        'start': i * char_duration,
                        'end': (i + 1) * char_duration,
                        'score': 0.5
                    })
        
        return char_timestamps
    
    def _align_with_expected_sequence(
        self,
        char_timestamps: List[Dict],
        pinyin_sequence: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """将识别结果与期望序列对齐。"""
        aligned_results = []
        
        expected_chars = [item['char'] for item in pinyin_sequence]
        recognized_chars = [item['char'] for item in char_timestamps]
        
        # 简单匹配
        for i, expected in enumerate(pinyin_sequence):
            char = expected['char']
            pinyin = expected['pinyin']
            
            # 在识别结果中查找
            matched = None
            for j, ts in enumerate(char_timestamps):
                if ts['char'] == char:
                    matched = ts
                    break
            
            if matched:
                aligned_results.append({
                    'char': char,
                    'pinyin': pinyin,
                    'start': matched['start'],
                    'end': matched['end'],
                    'score': matched['score']
                })
            else:
                # 未匹配，估计时间戳
                ts = self._estimate_timestamp(
                    i, len(pinyin_sequence), char_timestamps, audio_duration
                )
                aligned_results.append({
                    'char': char,
                    'pinyin': pinyin,
                    'start': ts['start'],
                    'end': ts['end'],
                    'score': 0.3
                })
        
        return aligned_results
    
    def _estimate_timestamp(
        self,
        index: int,
        total_chars: int,
        char_timestamps: List[Dict],
        audio_duration: float
    ) -> Dict:
        """估计缺失字符的时间戳。"""
        if char_timestamps:
            all_starts = [ts['start'] for ts in char_timestamps]
            all_ends = [ts['end'] for ts in char_timestamps]
            total_start = min(all_starts)
            total_end = max(all_ends)
            char_duration = (total_end - total_start) / max(1, total_chars)
            return {
                'start': total_start + index * char_duration,
                'end': total_start + (index + 1) * char_duration
            }
        
        char_duration = audio_duration / max(1, total_chars)
        return {
            'start': index * char_duration,
            'end': (index + 1) * char_duration
        }
    
    def _postprocess_timestamps(
        self,
        aligned_results: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """后处理时间戳。"""
        if not aligned_results:
            return aligned_results
        
        processed = []
        
        for i, item in enumerate(aligned_results):
            start = max(0, item['start'])
            end = max(0, item['end'])
            
            if end <= start:
                end = start + self.MIN_CHAR_DURATION
            
            duration = end - start
            if duration < self.MIN_CHAR_DURATION:
                end = start + self.MIN_CHAR_DURATION
            elif duration > self.MAX_CHAR_DURATION:
                end = start + self.MAX_CHAR_DURATION
            
            if end > audio_duration:
                end = audio_duration
                if start >= end:
                    start = max(0, end - self.MIN_CHAR_DURATION)
            
            processed.append({
                **item,
                'start': start,
                'end': end
            })
        
        return processed
    
    def _create_fallback_alignment(
        self,
        pinyin_sequence: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """创建备用对齐（均匀分配）。"""
        n_chars = len(pinyin_sequence)
        if n_chars == 0:
            return []
        
        char_duration = audio_duration / n_chars
        char_duration = max(self.MIN_CHAR_DURATION, min(char_duration, self.MAX_CHAR_DURATION))
        
        results = []
        for i, item in enumerate(pinyin_sequence):
            results.append({
                'char': item['char'],
                'pinyin': item['pinyin'],
                'start': i * char_duration,
                'end': (i + 1) * char_duration,
                'score': 0.3
            })
        
        return results
    
    def is_available(self) -> bool:
        """检查是否可用。"""
        return self.funasr_available
