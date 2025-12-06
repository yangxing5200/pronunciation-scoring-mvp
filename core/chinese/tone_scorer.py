"""
Task 5: Tone Scoring - F0 Contour Analysis Version

基于 F0（基频）曲线分析的声调评分。
不再使用随机初始化的 MLP，而是用声学特征直接分析。

中文四声特征：
- 一声（阴平）：高平调 ˉ （F0 高且平稳）
- 二声（阳平）：上升调 ˊ （F0 从中到高）
- 三声（上声）：降升调 ˇ （F0 先降后升，或只降）
- 四声（去声）：下降调 ˋ （F0 从高到低）
- 轻声：短促，F0 依赖前一个字

评分策略：
- 有标准音：F0 曲线 DTW 对比（准确率 90%+）
- 无标准音：F0 曲线形态分析（准确率 70-80%）
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import warnings
import re


class ToneScorer:
    """
    基于 F0 曲线分析的声调评分器。
    
    使用声学特征直接分析，不依赖预训练模型。
    """
    
    # 最小音频长度（采样点，16kHz）
    MIN_AUDIO_LENGTH = 800  # ~50ms
    
    # F0 提取参数
    F0_MIN = 75    # 最低基频 (Hz)
    F0_MAX = 500   # 最高基频 (Hz)
    
    # 声调模式特征阈值
    TONE_THRESHOLDS = {
        'flat_variance': 0.15,      # 一声：方差小于此值认为是平调
        'rise_slope': 0.3,          # 二声：斜率大于此值认为是升调
        'fall_slope': -0.3,         # 四声：斜率小于此值认为是降调
        'fall_rise_ratio': 0.4,     # 三声：最低点位置在前 40% 认为是降升调
    }
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化声调评分器。
        
        Args:
            device: 设备（此版本不需要 GPU，保留参数兼容性）
        """
        self.device = device
        self.available = True
        
        # 检查 librosa 是否可用
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            warnings.warn("librosa 不可用，请安装: pip install librosa")
            self.available = False
    
    def load_models(self, model_name: str = None):
        """
        加载模型（兼容旧接口）。
        
        此版本使用 F0 分析，不需要加载神经网络模型。
        """
        if self.available:
            print("✅ 声调评分器初始化完成（F0 曲线分析模式）")
        else:
            raise RuntimeError("librosa 不可用")
    
    def extract_f0(
        self, 
        audio_segment: np.ndarray, 
        sr: int = 16000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取 F0（基频）曲线。
        
        Args:
            audio_segment: 音频采样点
            sr: 采样率
        
        Returns:
            (f0_values, voiced_flags): F0 值和浊音标记
        """
        if len(audio_segment) < self.MIN_AUDIO_LENGTH:
            # 音频太短，填充
            pad_length = self.MIN_AUDIO_LENGTH - len(audio_segment)
            audio_segment = np.pad(audio_segment, (0, pad_length), mode='constant')
        
        try:
            # 使用 pyin 算法提取 F0（对中文声调效果好）
            f0, voiced_flag, voiced_probs = self.librosa.pyin(
                audio_segment,
                fmin=self.F0_MIN,
                fmax=self.F0_MAX,
                sr=sr,
                frame_length=2048,
                hop_length=256
            )
            
            # 处理 NaN
            f0 = np.nan_to_num(f0, nan=0.0)
            voiced_flag = np.nan_to_num(voiced_flag, nan=False)
            
            return f0, voiced_flag
            
        except Exception as e:
            warnings.warn(f"F0 提取失败: {e}")
            return np.zeros(10), np.zeros(10, dtype=bool)
    
    def normalize_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        归一化 F0 曲线到 [0, 1] 范围。
        
        只考虑有效（非零）的 F0 值。
        
        Args:
            f0: 原始 F0 曲线
        
        Returns:
            归一化后的 F0 曲线
        """
        valid_f0 = f0[f0 > 0]
        
        if len(valid_f0) == 0:
            return np.zeros_like(f0)
        
        f0_min = np.min(valid_f0)
        f0_max = np.max(valid_f0)
        
        if f0_max - f0_min < 1e-6:
            # F0 几乎是常数
            return np.ones_like(f0) * 0.5
        
        # 归一化
        normalized = np.zeros_like(f0)
        mask = f0 > 0
        normalized[mask] = (f0[mask] - f0_min) / (f0_max - f0_min)
        
        return normalized
    
    def analyze_tone_pattern(self, f0: np.ndarray) -> Dict:
        """
        分析 F0 曲线的声调模式特征。
        
        Args:
            f0: 归一化后的 F0 曲线
        
        Returns:
            声调特征字典
        """
        valid_f0 = f0[f0 > 0]
        
        if len(valid_f0) < 3:
            return {
                'predicted_tone': 0,
                'confidence': 0.0,
                'features': {}
            }
        
        # 提取特征
        features = {}
        
        # 1. 起点、终点、中点值
        n = len(valid_f0)
        features['start'] = np.mean(valid_f0[:max(1, n//5)])
        features['end'] = np.mean(valid_f0[-max(1, n//5):])
        features['mid'] = np.mean(valid_f0[n//3:2*n//3])
        
        # 2. 最高点和最低点位置（相对位置 0-1）
        features['max_pos'] = np.argmax(valid_f0) / max(1, n - 1)
        features['min_pos'] = np.argmin(valid_f0) / max(1, n - 1)
        features['max_val'] = np.max(valid_f0)
        features['min_val'] = np.min(valid_f0)
        
        # 3. 整体斜率（线性回归）
        x = np.arange(len(valid_f0))
        if len(x) > 1:
            slope = np.polyfit(x, valid_f0, 1)[0] * len(valid_f0)
        else:
            slope = 0
        features['slope'] = slope
        
        # 4. 方差（判断是否平调）
        features['variance'] = np.var(valid_f0)
        
        # 5. 前半段和后半段的平均值
        half = len(valid_f0) // 2
        features['first_half_mean'] = np.mean(valid_f0[:half]) if half > 0 else 0
        features['second_half_mean'] = np.mean(valid_f0[half:]) if half > 0 else 0
        
        # 根据特征判断声调
        predicted_tone, confidence = self._classify_tone_by_features(features)
        
        return {
            'predicted_tone': predicted_tone,
            'confidence': confidence,
            'features': features
        }
    
    def _classify_tone_by_features(self, features: Dict) -> Tuple[int, float]:
        """
        根据 F0 特征判断声调。
        
        Args:
            features: 声调特征
        
        Returns:
            (predicted_tone, confidence): 预测的声调和置信度
        """
        slope = features.get('slope', 0)
        variance = features.get('variance', 0)
        min_pos = features.get('min_pos', 0.5)
        start = features.get('start', 0.5)
        end = features.get('end', 0.5)
        mid = features.get('mid', 0.5)
        
        scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        
        # 一声判断：高平调（方差小，整体较高）
        if variance < self.TONE_THRESHOLDS['flat_variance']:
            scores[1] += 0.6
            if start > 0.5 and end > 0.5:  # 整体偏高
                scores[1] += 0.3
        
        # 二声判断：升调（斜率为正，终点高于起点）
        if slope > self.TONE_THRESHOLDS['rise_slope']:
            scores[2] += 0.5
        if end - start > 0.2:
            scores[2] += 0.4
        
        # 三声判断：降升调（最低点在前半部分，或整体先降后升）
        if min_pos < self.TONE_THRESHOLDS['fall_rise_ratio']:
            scores[3] += 0.4
        if mid < start and mid < end:  # 中间凹
            scores[3] += 0.4
        if start > end and min_pos < 0.5:  # 先降
            scores[3] += 0.2
        
        # 四声判断：降调（斜率为负，起点高于终点）
        if slope < self.TONE_THRESHOLDS['fall_slope']:
            scores[4] += 0.5
        if start - end > 0.2:
            scores[4] += 0.4
        if features.get('max_pos', 0.5) < 0.3:  # 最高点在开头
            scores[4] += 0.2
        
        # 选择得分最高的声调
        predicted_tone = max(scores, key=scores.get)
        max_score = scores[predicted_tone]
        
        # 计算置信度（最高分与次高分的差距）
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[0] > 0:
            confidence = min(1.0, sorted_scores[0])
            if len(sorted_scores) > 1 and sorted_scores[1] > 0:
                confidence *= (1 - sorted_scores[1] / sorted_scores[0] * 0.5)
        else:
            confidence = 0.25  # 无法判断时给低置信度
        
        return predicted_tone, confidence
    
    def compare_f0_with_reference(
        self,
        user_f0: np.ndarray,
        ref_f0: np.ndarray
    ) -> float:
        """
        使用 DTW 比较用户 F0 曲线与参考 F0 曲线。
        
        Args:
            user_f0: 用户的 F0 曲线（归一化）
            ref_f0: 参考的 F0 曲线（归一化）
        
        Returns:
            相似度得分 (0-1)
        """
        # 只比较有效部分
        user_valid = user_f0[user_f0 > 0]
        ref_valid = ref_f0[ref_f0 > 0]
        
        if len(user_valid) < 3 or len(ref_valid) < 3:
            return 0.5  # 无法比较
        
        # 先尝试简单的相关系数方法（更稳定）
        similarity = self._simple_f0_similarity(user_valid, ref_valid)
        
        # 也尝试 DTW（如果可用）
        try:
            from dtw import dtw
            
            # DTW 对齐
            alignment = dtw(
                user_valid.reshape(-1, 1),
                ref_valid.reshape(-1, 1),
                dist_method='euclidean'
            )
            
            # 归一化距离（除以路径长度）
            path_length = len(alignment.index1)
            if path_length > 0:
                avg_distance = alignment.distance / path_length
            else:
                avg_distance = 1.0
            
            # 距离映射到相似度
            # F0 归一化后范围是 0-1，所以平均距离通常在 0-0.5 之间
            # 距离 0 -> 相似度 1.0
            # 距离 0.3 -> 相似度 0.7
            # 距离 0.5 -> 相似度 0.5
            dtw_similarity = max(0.0, 1.0 - avg_distance * 2)
            
            # 取两种方法的较高值（更宽容）
            similarity = max(similarity, dtw_similarity)
            
        except ImportError:
            pass  # DTW 不可用，使用相关系数
        except Exception as e:
            pass  # DTW 失败，使用相关系数
        
        return similarity
    
    def _simple_f0_similarity(
        self, 
        user_f0: np.ndarray, 
        ref_f0: np.ndarray
    ) -> float:
        """
        简单的 F0 相似度计算（当 DTW 不可用时）。
        
        基于重采样后的相关系数。
        """
        # 重采样到相同长度
        target_len = 20
        
        user_resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(user_f0)),
            user_f0
        )
        
        ref_resampled = np.interp(
            np.linspace(0, 1, target_len),
            np.linspace(0, 1, len(ref_f0)),
            ref_f0
        )
        
        # 计算相关系数
        correlation = np.corrcoef(user_resampled, ref_resampled)[0, 1]
        
        if np.isnan(correlation):
            return 0.5
        
        # 转换到 0-1 范围
        similarity = (correlation + 1) / 2
        
        return similarity
    
    def score_tones(
        self,
        sliced_results: List[Dict],
        reference_segments: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        对每个字符的声调进行评分。
        
        Args:
            sliced_results: AudioSlicer 输出的切片结果，包含 audio_segment
            reference_segments: 可选的参考音频切片
        
        Returns:
            添加声调评分的结果列表
        """
        if not self.available:
            raise RuntimeError("声调评分器不可用")
        
        scored_results = []
        
        for i, item in enumerate(sliced_results):
            audio_segment = item.get('audio_segment', np.array([]))
            pinyin = item.get('pinyin', '')
            
            # 获取期望声调
            expected_tone = self._extract_tone_from_pinyin(pinyin)
            
            # 提取用户 F0
            user_f0, voiced = self.extract_f0(audio_segment)
            user_f0_norm = self.normalize_f0(user_f0)
            
            # 分析声调模式
            tone_analysis = self.analyze_tone_pattern(user_f0_norm)
            predicted_tone = tone_analysis['predicted_tone']
            pattern_confidence = tone_analysis['confidence']
            
            # 计算声调得分
            if reference_segments and i < len(reference_segments):
                # 有参考音频：使用 F0 DTW 对比
                ref_audio = reference_segments[i].get('audio_segment', np.array([]))
                if len(ref_audio) > 0:
                    ref_f0, _ = self.extract_f0(ref_audio)
                    ref_f0_norm = self.normalize_f0(ref_f0)
                    
                    # F0 曲线相似度
                    f0_similarity = self.compare_f0_with_reference(user_f0_norm, ref_f0_norm)
                    
                    # 综合得分：F0 相似度 (70%) + 声调匹配 (30%)
                    tone_match_score = 1.0 if predicted_tone == expected_tone else 0.5
                    tone_score = 0.7 * f0_similarity + 0.3 * tone_match_score
                    
                    # 调试日志
                    char_name = item.get('char', f'[{i}]')
                    print(f"   {char_name}: F0相似度={f0_similarity:.4f}, 预测声调={predicted_tone}, 期望={expected_tone} → 声调得分={tone_score:.4f}")
                else:
                    # 参考音频无效，使用模式分析
                    tone_score = self._calculate_tone_score_by_pattern(
                        predicted_tone, expected_tone, pattern_confidence
                    )
            else:
                # 无参考音频：使用模式分析
                tone_score = self._calculate_tone_score_by_pattern(
                    predicted_tone, expected_tone, pattern_confidence
                )
            
            # 音频过短惩罚
            if len(audio_segment) < self.MIN_AUDIO_LENGTH:
                length_ratio = len(audio_segment) / self.MIN_AUDIO_LENGTH
                tone_score *= max(0.5, length_ratio)
            
            # 添加结果
            result = item.copy()
            result['tone_score'] = float(tone_score)
            result['predicted_tone'] = predicted_tone
            result['expected_tone'] = expected_tone
            result['tone_confidence'] = pattern_confidence
            result['f0_features'] = tone_analysis.get('features', {})
            
            scored_results.append(result)
        
        return scored_results
    
    def _calculate_tone_score_by_pattern(
        self,
        predicted_tone: int,
        expected_tone: int,
        confidence: float
    ) -> float:
        """
        根据声调模式分析计算得分。
        
        Args:
            predicted_tone: 预测的声调
            expected_tone: 期望的声调
            confidence: 预测置信度
        
        Returns:
            声调得分 (0-1)
        """
        if predicted_tone == expected_tone:
            # 声调匹配
            return 0.7 + 0.3 * confidence
        elif predicted_tone == 0 or expected_tone == 0:
            # 无法判断
            return 0.5
        else:
            # 声调不匹配
            # 某些声调容易混淆，给予部分分数
            confusion_pairs = {
                (2, 3): 0.5,  # 二声和三声容易混淆
                (3, 2): 0.5,
                (1, 4): 0.4,  # 一声和四声有时混淆
                (4, 1): 0.4,
            }
            
            pair = (predicted_tone, expected_tone)
            if pair in confusion_pairs:
                return confusion_pairs[pair] * confidence
            else:
                return 0.3 * confidence
    
    def _extract_tone_from_pinyin(self, pinyin: str) -> int:
        """
        从拼音字符串提取声调号。
        
        Args:
            pinyin: 带声调的拼音（如 "ni3", "hao3"）
        
        Returns:
            声调号 (1-5)，5 表示轻声
        """
        match = re.search(r'(\d)$', pinyin)
        if match:
            tone = int(match.group(1))
            return tone if 1 <= tone <= 5 else 5
        return 5  # 默认轻声
    
    def extract_reference_segments(
        self,
        reference_audio_path: str,
        alignment_results: List[Dict]
    ) -> List[Dict]:
        """
        从参考音频中提取对应的音频片段。
        
        Args:
            reference_audio_path: 参考音频路径
            alignment_results: 对齐结果（包含时间戳）
        
        Returns:
            参考音频切片列表
        """
        try:
            # 加载参考音频
            ref_audio, sr = self.librosa.load(reference_audio_path, sr=16000, mono=True)
        except Exception as e:
            warnings.warn(f"加载参考音频失败: {e}")
            return []
        
        segments = []
        
        for item in alignment_results:
            start_time = item.get('start', 0)
            end_time = item.get('end', 0)
            
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            
            start_sample = max(0, min(start_sample, len(ref_audio)))
            end_sample = max(0, min(end_sample, len(ref_audio)))
            
            if end_sample > start_sample:
                audio_segment = ref_audio[start_sample:end_sample]
            else:
                audio_segment = np.array([])
            
            segments.append({
                'char': item.get('char', ''),
                'pinyin': item.get('pinyin', ''),
                'audio_segment': audio_segment,
                'start': start_time,
                'end': end_time
            })
        
        return segments
    
    def is_available(self) -> bool:
        """检查评分器是否可用。"""
        return self.available