"""
Task 2: Audio Alignment (Forced Alignment) - FunASR Version

使用 FunASR (阿里达摩院) 进行中文语音识别和字级别对齐。
FunASR 的 Paraformer 模型对中文识别准确率远高于 WhisperX。

优点：
- 中文识别准确率高（阿里海量中文数据训练）
- 原生支持字级别时间戳
- 支持 VAD（语音活动检测）
- 离线部署，模型自动下载到本地

安装: pip install funasr modelscope
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
import numpy as np
import re


class ChineseAudioAligner:
    """
    使用 FunASR 进行中文音频对齐。
    
    提供字级别时间戳，用于后续的音频切片和发音评分。
    """
    
    # 字符段最小/最大时长（秒）
    MIN_CHAR_DURATION = 0.05  # 50ms
    MAX_CHAR_DURATION = 1.0   # 1s
    DEFAULT_CHAR_DURATION = 0.25  # 250ms（默认估计值）
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化音频对齐器。
        
        Args:
            device: 使用的设备 ('cuda' 或 'cpu')，None 时自动检测
        """
        self.device = device
        self.funasr_available = False
        self.whisperx_available = False
        self.funasr_model = None
        self.whisperx_model = None
        self.align_model = None
        self.align_metadata = None
        
        # 自动检测设备
        try:
            import torch
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🖥️ 使用设备: {self.device}")
        except ImportError:
            self.device = "cpu"
        
        # 检测 FunASR（首选）
        try:
            from funasr import AutoModel
            self.AutoModel = AutoModel
            self.funasr_available = True
            print("✅ FunASR 可用 - 推荐用于中文识别")
        except ImportError:
            warnings.warn(
                "FunASR 不可用。安装命令: pip install funasr modelscope\n"
                "FunASR 的中文识别准确率远高于 WhisperX。"
            )
        
        # 检测 WhisperX（备选）
        try:
            import whisperx
            self.whisperx = whisperx
            self.whisperx_available = True
            if not self.funasr_available:
                print("⚠️ 使用 WhisperX 作为备选（中文准确率较低）")
        except ImportError:
            if not self.funasr_available:
                warnings.warn(
                    "FunASR 和 WhisperX 都不可用。\n"
                    "请安装 FunASR: pip install funasr modelscope"
                )
    
    def load_models(self, model_size: str = "base"):
        """
        加载 ASR 模型。
        
        Args:
            model_size: 模型大小（仅对 WhisperX 有效）
        """
        # 优先使用 FunASR
        if self.funasr_available:
            try:
                print("📥 加载 FunASR Paraformer 模型（首次运行会自动下载）...")
                
                # Paraformer-zh: 阿里开源的中文语音识别模型
                # - 支持字级别时间戳
                # - 识别准确率高
                # - 显存占用约 2-3GB
                self.funasr_model = self.AutoModel(
                    model="paraformer-zh",           # 中文 Paraformer
                    model_revision="v2.0.4",         # 稳定版本
                    vad_model="fsmn-vad",            # 语音活动检测
                    vad_model_revision="v2.0.4",
                    punc_model="ct-punc",            # 标点恢复
                    punc_model_revision="v2.0.4",
                    device=self.device
                )
                
                print(f"✅ FunASR Paraformer 模型加载完成 (设备: {self.device})")
                return
                
            except Exception as e:
                warnings.warn(f"FunASR 加载失败: {e}")
                self.funasr_available = False
                print("⚠️ 尝试使用 WhisperX 作为备选...")
        
        # 备选: WhisperX
        if self.whisperx_available:
            try:
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.whisperx_model = self.whisperx.load_model(
                    model_size,
                    self.device,
                    compute_type=compute_type
                )
                
                # 加载中文对齐模型
                self.align_model, self.align_metadata = self.whisperx.load_align_model(
                    language_code="zh",
                    device=self.device
                )
                
                print(f"⚠️ WhisperX 模型加载完成 (设备: {self.device})")
                print("   注意: WhisperX 中文准确率较低，建议安装 FunASR")
                
            except Exception as e:
                raise RuntimeError(f"无法加载任何 ASR 模型: {e}")
        else:
            raise RuntimeError(
                "没有可用的 ASR 模型。请安装 FunASR:\n"
                "pip install funasr modelscope"
            )
    
    def align_audio(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        对齐音频与拼音序列，获取字级别时间戳。
        
        Args:
            audio_path: 音频文件路径
            pinyin_sequence: PinyinMapper 输出的拼音序列
                            [{'char': '你', 'pinyin': 'ni3'}, ...]
        
        Returns:
            对齐结果列表:
            [
                {"char": "你", "pinyin": "ni3", "start": 0.12, "end": 0.36, "score": 0.95},
                {"char": "好", "pinyin": "hao3", "start": 0.36, "end": 0.58, "score": 0.92},
                ...
            ]
        """
        if self.funasr_model is not None:
            return self._align_with_funasr(audio_path, pinyin_sequence)
        elif self.whisperx_model is not None:
            return self._align_with_whisperx(audio_path, pinyin_sequence)
        else:
            raise RuntimeError("模型未加载。请先调用 load_models()")
    
    def _align_with_funasr(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        使用 FunASR 进行对齐 - 提供精确的字级别时间戳。
        
        Args:
            audio_path: 音频文件路径
            pinyin_sequence: 期望的拼音序列
        
        Returns:
            对齐结果列表
        """
        # 获取音频时长
        audio_duration = self._get_audio_duration(audio_path)
        
        # FunASR 识别
        try:
            result = self.funasr_model.generate(
                input=audio_path,
                batch_size_s=300,          # 批处理大小（秒）
                return_raw_text=False,     # 返回结构化结果
            )
        except Exception as e:
            warnings.warn(f"FunASR 识别失败: {e}")
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        # 解析 FunASR 结果
        if not result or len(result) == 0:
            warnings.warn("FunASR 返回空结果")
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        # FunASR 返回格式: [{'text': '你好', 'timestamp': [[0, 250], [250, 500]], ...}]
        funasr_result = result[0] if isinstance(result, list) else result
        
        recognized_text = funasr_result.get('text', '')
        timestamps = funasr_result.get('timestamp', [])
        
        print(f"🎤 FunASR 识别结果: {recognized_text}")
        print(f"   期望文本: {''.join([item['char'] for item in pinyin_sequence])}")
        print(f"   🔍 时间戳数量: {len(timestamps)}, 识别文本长度: {len(recognized_text)}")
        
        # 关键修复：保留每个字符（含标点）与时间戳的对应，然后只提取汉字的时间戳
        char_timestamps = self._build_char_timestamps_with_punctuation(
            recognized_text,
            timestamps,
            audio_duration
        )
        
        print(f"   🔍 提取到的汉字时间戳数量: {len(char_timestamps)}")
        
        # 将识别结果与期望序列对齐
        aligned_results = self._align_with_expected_sequence(
            char_timestamps,
            pinyin_sequence,
            audio_duration
        )
        
        # 后处理：确保时间戳有效
        aligned_results = self._postprocess_timestamps(aligned_results, audio_duration)
        
        return aligned_results
    
    def _build_char_timestamps_with_punctuation(
        self,
        recognized_text: str,
        timestamps: List,
        audio_duration: float
    ) -> List[Dict]:
        """
        从带标点的识别结果中提取汉字的时间戳。
        
        FunASR 的 timestamp 是针对完整文本（含标点）的，
        需要先建立字符-时间戳对应，再筛选出汉字。
        
        Args:
            recognized_text: 识别到的完整文本（含标点）
            timestamps: FunASR 返回的时间戳 [[start_ms, end_ms], ...]
            audio_duration: 音频总时长（秒）
        
        Returns:
            仅包含汉字的时间戳列表
        """
        char_timestamps = []
        
        # 检查时间戳数量是否与文本长度匹配
        if timestamps and len(timestamps) == len(recognized_text):
            # 完美匹配：每个字符（含标点）都有时间戳
            for i, char in enumerate(recognized_text):
                # 只保留汉字
                if re.match(r'[\u4e00-\u9fff]', char):
                    ts = timestamps[i]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start = ts[0] / 1000.0  # 毫秒转秒
                        end = ts[1] / 1000.0
                    else:
                        # 时间戳格式异常
                        continue
                    
                    char_timestamps.append({
                        'char': char,
                        'start': start,
                        'end': end,
                        'score': 0.9
                    })
            
            print(f"   ✅ 从 {len(recognized_text)} 个字符中提取了 {len(char_timestamps)} 个汉字时间戳")
        
        elif timestamps:
            # 时间戳数量不匹配，尝试其他策略
            print(f"   ⚠️ 时间戳数量 ({len(timestamps)}) 与文本长度 ({len(recognized_text)}) 不匹配")
            
            # 提取纯汉字
            chinese_chars = self._extract_chinese_chars(recognized_text)
            
            # 如果时间戳数量等于汉字数量，直接对应
            if len(timestamps) == len(chinese_chars):
                print(f"   ✅ 时间戳数量与汉字数量匹配")
                for i, char in enumerate(chinese_chars):
                    ts = timestamps[i]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        start = ts[0] / 1000.0
                        end = ts[1] / 1000.0
                        char_timestamps.append({
                            'char': char,
                            'start': start,
                            'end': end,
                            'score': 0.85
                        })
            else:
                # 使用备用策略：基于时间范围均匀分配
                print(f"   ⚠️ 使用均匀分配策略")
                char_timestamps = self._distribute_timestamps(
                    chinese_chars, timestamps, audio_duration
                )
        else:
            # 没有时间戳
            print(f"   ⚠️ 无时间戳信息，使用均匀分配")
            chinese_chars = self._extract_chinese_chars(recognized_text)
            char_duration = audio_duration / max(1, len(chinese_chars))
            for i, char in enumerate(chinese_chars):
                char_timestamps.append({
                    'char': char,
                    'start': i * char_duration,
                    'end': (i + 1) * char_duration,
                    'score': 0.5
                })
        
        return char_timestamps
    
    def _build_char_timestamps(
        self,
        recognized_chars: List[str],
        timestamps: List,
        audio_duration: float
    ) -> List[Dict]:
        """
        构建字符-时间戳映射。
        
        Args:
            recognized_chars: 识别到的中文字符列表
            timestamps: FunASR 返回的时间戳 [[start_ms, end_ms], ...]
            audio_duration: 音频总时长（秒）
        
        Returns:
            字符时间戳列表
        """
        char_timestamps = []
        
        # FunASR 时间戳是毫秒，需要转换为秒
        if timestamps and len(timestamps) == len(recognized_chars):
            # 时间戳数量与字符数量匹配
            for i, char in enumerate(recognized_chars):
                ts = timestamps[i]
                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                    start = ts[0] / 1000.0  # 毫秒转秒
                    end = ts[1] / 1000.0
                else:
                    # 时间戳格式异常，使用估计值
                    start = i * self.DEFAULT_CHAR_DURATION
                    end = (i + 1) * self.DEFAULT_CHAR_DURATION
                
                char_timestamps.append({
                    'char': char,
                    'start': start,
                    'end': end,
                    'score': 0.9  # FunASR 识别置信度较高
                })
        
        elif timestamps and len(timestamps) > 0:
            # 时间戳数量与字符数量不匹配，尝试智能分配
            # 这种情况可能是 FunASR 返回的是词级别时间戳
            total_chars = len(recognized_chars)
            
            if len(timestamps) == 1:
                # 只有一个时间戳段，平均分配
                ts = timestamps[0]
                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                    segment_start = ts[0] / 1000.0
                    segment_end = ts[1] / 1000.0
                else:
                    segment_start = 0
                    segment_end = audio_duration
                
                char_duration = (segment_end - segment_start) / max(1, total_chars)
                
                for i, char in enumerate(recognized_chars):
                    char_timestamps.append({
                        'char': char,
                        'start': segment_start + i * char_duration,
                        'end': segment_start + (i + 1) * char_duration,
                        'score': 0.7
                    })
            else:
                # 多个时间戳段，尝试按比例分配
                char_timestamps = self._distribute_timestamps(
                    recognized_chars, timestamps, audio_duration
                )
        
        else:
            # 没有时间戳，平均分配
            char_duration = audio_duration / max(1, len(recognized_chars))
            for i, char in enumerate(recognized_chars):
                char_timestamps.append({
                    'char': char,
                    'start': i * char_duration,
                    'end': (i + 1) * char_duration,
                    'score': 0.5
                })
        
        return char_timestamps
    
    def _distribute_timestamps(
        self,
        chars: List[str],
        timestamps: List,
        audio_duration: float
    ) -> List[Dict]:
        """
        将多个时间戳段分配给字符列表。
        
        处理 FunASR 返回词级别时间戳的情况。
        """
        char_timestamps = []
        
        # 展平所有时间戳
        all_starts = []
        all_ends = []
        
        for ts in timestamps:
            if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                all_starts.append(ts[0] / 1000.0)
                all_ends.append(ts[1] / 1000.0)
        
        if not all_starts:
            # 无有效时间戳，平均分配
            char_duration = audio_duration / max(1, len(chars))
            for i, char in enumerate(chars):
                char_timestamps.append({
                    'char': char,
                    'start': i * char_duration,
                    'end': (i + 1) * char_duration,
                    'score': 0.5
                })
            return char_timestamps
        
        # 计算总时间范围
        total_start = min(all_starts)
        total_end = max(all_ends)
        total_duration = total_end - total_start
        
        # 按字符数平均分配
        char_duration = total_duration / max(1, len(chars))
        
        for i, char in enumerate(chars):
            char_timestamps.append({
                'char': char,
                'start': total_start + i * char_duration,
                'end': total_start + (i + 1) * char_duration,
                'score': 0.6
            })
        
        return char_timestamps
    
    def _align_with_expected_sequence(
        self,
        char_timestamps: List[Dict],
        pinyin_sequence: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """
        将识别结果与期望的拼音序列对齐。
        
        使用动态规划找到最优匹配。
        
        Args:
            char_timestamps: 识别到的字符及时间戳
            pinyin_sequence: 期望的拼音序列
            audio_duration: 音频时长
        
        Returns:
            对齐后的结果
        """
        expected_chars = [item['char'] for item in pinyin_sequence]
        pinyin_map = {item['char']: item['pinyin'] for item in pinyin_sequence}
        
        recognized_chars = [item['char'] for item in char_timestamps]
        
        # 如果完全匹配，直接使用
        if recognized_chars == expected_chars:
            print("✅ 识别结果与期望完全匹配")
            aligned_results = []
            for i, item in enumerate(pinyin_sequence):
                ts = char_timestamps[i]
                aligned_results.append({
                    'char': item['char'],
                    'pinyin': item['pinyin'],
                    'start': ts['start'],
                    'end': ts['end'],
                    'score': ts['score']
                })
            return aligned_results
        
        # 使用编辑距离对齐
        print(f"⚠️ 识别结果与期望不完全匹配，使用对齐算法...")
        print(f"   识别: {''.join(recognized_chars)}")
        print(f"   期望: {''.join(expected_chars)}")
        
        aligned_results = self._dtw_align(
            char_timestamps,
            pinyin_sequence,
            audio_duration
        )
        
        return aligned_results
    
    def _dtw_align(
        self,
        char_timestamps: List[Dict],
        pinyin_sequence: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """
        使用 DTW（动态时间规整）对齐识别结果和期望序列。
        
        Args:
            char_timestamps: 识别到的字符时间戳
            pinyin_sequence: 期望的拼音序列
            audio_duration: 音频时长
        
        Returns:
            对齐后的结果（保证与 pinyin_sequence 长度相同）
        """
        n_recognized = len(char_timestamps)
        n_expected = len(pinyin_sequence)
        
        if n_recognized == 0:
            # 没有识别到任何内容，使用均匀分配
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
        
        # 构建相似度矩阵
        # 1 = 完全匹配, 0 = 不匹配
        similarity_matrix = np.zeros((n_expected, n_recognized))
        
        for i, expected_item in enumerate(pinyin_sequence):
            expected_char = expected_item['char']
            for j, recognized_item in enumerate(char_timestamps):
                recognized_char = recognized_item['char']
                if expected_char == recognized_char:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 可以扩展：使用拼音相似度等
                    similarity_matrix[i, j] = 0.0
        
        # 使用贪婪匹配找到每个期望字符的最佳匹配
        aligned_results = []
        used_indices = set()
        
        for i, expected_item in enumerate(pinyin_sequence):
            expected_char = expected_item['char']
            pinyin = expected_item['pinyin']
            
            # 找到最佳匹配（优先精确匹配，其次位置相近）
            best_match_idx = -1
            best_score = -1
            
            for j in range(n_recognized):
                if j in used_indices:
                    continue
                
                sim = similarity_matrix[i, j]
                
                # 考虑位置因素（期望位置相近的优先）
                expected_position_ratio = i / max(1, n_expected - 1) if n_expected > 1 else 0.5
                actual_position_ratio = j / max(1, n_recognized - 1) if n_recognized > 1 else 0.5
                position_penalty = abs(expected_position_ratio - actual_position_ratio)
                
                # 综合得分 = 相似度 - 位置惩罚
                score = sim - position_penalty * 0.3
                
                if score > best_score:
                    best_score = score
                    best_match_idx = j
            
            if best_match_idx >= 0 and similarity_matrix[i, best_match_idx] > 0.5:
                # 找到匹配
                used_indices.add(best_match_idx)
                ts = char_timestamps[best_match_idx]
                aligned_results.append({
                    'char': expected_char,
                    'pinyin': pinyin,
                    'start': ts['start'],
                    'end': ts['end'],
                    'score': ts['score'] * similarity_matrix[i, best_match_idx]
                })
            else:
                # 没有找到匹配，使用插值
                interpolated_ts = self._interpolate_timestamp(
                    i, n_expected, aligned_results, char_timestamps, audio_duration
                )
                aligned_results.append({
                    'char': expected_char,
                    'pinyin': pinyin,
                    'start': interpolated_ts['start'],
                    'end': interpolated_ts['end'],
                    'score': 0.3  # 低置信度
                })
        
        return aligned_results
    
    def _interpolate_timestamp(
        self,
        index: int,
        total_chars: int,
        previous_alignments: List[Dict],
        char_timestamps: List[Dict],
        audio_duration: float
    ) -> Dict:
        """
        为未匹配的字符插值时间戳。
        
        Args:
            index: 当前字符索引
            total_chars: 总字符数
            previous_alignments: 已对齐的结果
            char_timestamps: 所有识别到的时间戳
            audio_duration: 音频总时长
        
        Returns:
            插值的时间戳 {'start': float, 'end': float}
        """
        # 策略1: 基于前一个已对齐字符
        if previous_alignments:
            last_end = previous_alignments[-1]['end']
            remaining_duration = audio_duration - last_end
            remaining_chars = total_chars - index
            
            if remaining_chars > 0 and remaining_duration > 0:
                estimated_duration = min(
                    remaining_duration / remaining_chars,
                    self.MAX_CHAR_DURATION
                )
                estimated_duration = max(estimated_duration, self.MIN_CHAR_DURATION)
                
                return {
                    'start': last_end,
                    'end': last_end + estimated_duration
                }
        
        # 策略2: 基于识别到的时间戳范围
        if char_timestamps:
            all_starts = [ts['start'] for ts in char_timestamps]
            all_ends = [ts['end'] for ts in char_timestamps]
            
            total_start = min(all_starts) if all_starts else 0
            total_end = max(all_ends) if all_ends else audio_duration
            
            char_duration = (total_end - total_start) / max(1, total_chars)
            char_duration = max(self.MIN_CHAR_DURATION, min(char_duration, self.MAX_CHAR_DURATION))
            
            return {
                'start': total_start + index * char_duration,
                'end': total_start + (index + 1) * char_duration
            }
        
        # 策略3: 均匀分配
        char_duration = audio_duration / max(1, total_chars)
        char_duration = max(self.MIN_CHAR_DURATION, min(char_duration, self.MAX_CHAR_DURATION))
        
        return {
            'start': index * char_duration,
            'end': (index + 1) * char_duration
        }
    
    def _postprocess_timestamps(
        self,
        aligned_results: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """
        后处理时间戳，确保有效性。
        
        - 确保 start < end
        - 确保时长在合理范围内
        - 确保不超出音频边界
        - 修复重叠
        
        Args:
            aligned_results: 对齐结果
            audio_duration: 音频时长
        
        Returns:
            处理后的结果
        """
        if not aligned_results:
            return aligned_results
        
        processed = []
        
        for i, item in enumerate(aligned_results):
            start = item['start']
            end = item['end']
            
            # 确保不为负数
            start = max(0, start)
            end = max(0, end)
            
            # 确保 start < end
            if end <= start:
                end = start + self.MIN_CHAR_DURATION
            
            # 确保时长在合理范围
            duration = end - start
            if duration < self.MIN_CHAR_DURATION:
                end = start + self.MIN_CHAR_DURATION
            elif duration > self.MAX_CHAR_DURATION:
                end = start + self.MAX_CHAR_DURATION
            
            # 确保不超出音频边界
            if end > audio_duration:
                end = audio_duration
                if start >= end:
                    start = max(0, end - self.MIN_CHAR_DURATION)
            
            processed.append({
                **item,
                'start': start,
                'end': end
            })
        
        # 修复重叠：确保每个字符的 end <= 下一个字符的 start
        for i in range(len(processed) - 1):
            if processed[i]['end'] > processed[i + 1]['start']:
                # 有重叠，取中点
                mid = (processed[i]['start'] + processed[i + 1]['end']) / 2
                processed[i]['end'] = mid
                processed[i + 1]['start'] = mid
        
        return processed
    
    def _create_fallback_alignment(
        self,
        pinyin_sequence: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """
        创建备用对齐结果（均匀分配）。
        
        当 ASR 失败时使用。
        
        Args:
            pinyin_sequence: 期望的拼音序列
            audio_duration: 音频时长
        
        Returns:
            均匀分配的对齐结果
        """
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
                'score': 0.3  # 低置信度标记
            })
        
        return results
    
    def _extract_chinese_chars(self, text: str) -> List[str]:
        """
        从文本中提取中文字符。
        
        Args:
            text: 输入文本
        
        Returns:
            中文字符列表
        """
        return re.findall(r'[\u4e00-\u9fff]', text)
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        获取音频文件时长。
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            时长（秒）
        """
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            return len(audio) / sr
        except Exception as e:
            warnings.warn(f"无法获取音频时长: {e}")
            return 5.0  # 默认 5 秒
    
    # ========== WhisperX 备选实现 ==========
    
    def _align_with_whisperx(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        使用 WhisperX 进行对齐（备选方案）。
        
        注意: WhisperX 的中文对齐准确率较低。
        """
        if self.whisperx_model is None or self.align_model is None:
            raise RuntimeError("WhisperX 模型未加载")
        
        audio_duration = self._get_audio_duration(audio_path)
        
        try:
            # 加载音频
            audio = self.whisperx.load_audio(audio_path)
            
            # 转录
            result = self.whisperx_model.transcribe(
                audio,
                batch_size=16,
                language="zh"
            )
            
            # 对齐
            aligned_result = self.whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=True
            )
            
            # 提取字符时间戳
            char_timestamps = self._extract_whisperx_char_timestamps(aligned_result)
            
            # 与期望序列对齐
            aligned_results = self._align_with_expected_sequence(
                char_timestamps,
                pinyin_sequence,
                audio_duration
            )
            
            # 后处理
            aligned_results = self._postprocess_timestamps(aligned_results, audio_duration)
            
            return aligned_results
            
        except Exception as e:
            warnings.warn(f"WhisperX 对齐失败: {e}")
            return self._create_fallback_alignment(pinyin_sequence, audio_duration)
    
    def _extract_whisperx_char_timestamps(self, aligned_result: Dict) -> List[Dict]:
        """
        从 WhisperX 结果中提取字符时间戳。
        """
        char_timestamps = []
        
        for segment in aligned_result.get("segments", []):
            for word_info in segment.get("words", []):
                if "chars" in word_info:
                    for char_info in word_info["chars"]:
                        char = char_info.get("char", "").strip()
                        if char and re.match(r'[\u4e00-\u9fff]', char):
                            char_timestamps.append({
                                'char': char,
                                'start': char_info.get('start', 0.0),
                                'end': char_info.get('end', 0.0),
                                'score': char_info.get('score', 0.5)
                            })
                else:
                    # 没有字符级对齐，按词处理
                    word = word_info.get("word", "").strip()
                    start = word_info.get("start", 0.0)
                    end = word_info.get("end", 0.0)
                    
                    chinese_chars = self._extract_chinese_chars(word)
                    if chinese_chars:
                        char_duration = (end - start) / len(chinese_chars)
                        for i, char in enumerate(chinese_chars):
                            char_timestamps.append({
                                'char': char,
                                'start': start + i * char_duration,
                                'end': start + (i + 1) * char_duration,
                                'score': word_info.get('score', 0.5)
                            })
        
        return char_timestamps
    
    def is_available(self) -> bool:
        """检查是否有可用的 ASR 模型。"""
        return self.funasr_available or self.whisperx_available