"""
Task 4: Acoustic Scoring - Enhanced Version

使用 WavLM 嵌入向量和余弦相似度进行声学评分。
评估声母/韵母的发音质量。

增强功能：
- 优化与标准音的对比逻辑
- 改进无标准音时的启发式评分
- 更好的短音频处理
"""

from typing import List, Dict, Optional
import numpy as np
import warnings


class AcousticScorer:
    """
    基于 WavLM 嵌入的声学评分器。
    
    通过比较用户音频与参考音频的嵌入相似度来评估发音质量。
    """
    
    # WavLM 处理的最小音频长度
    MIN_AUDIO_LENGTH = 512  # 采样点 @ 16kHz (~32ms)
    
    # WavLM-base+ 的嵌入维度
    EMBEDDING_DIM = 768
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化声学评分器。
        
        Args:
            device: 使用的设备 ('cuda' 或 'cpu')，None 时自动检测
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
            warnings.warn("PyTorch 不可用")
            return
        
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMModel
            self.FeatureExtractor = Wav2Vec2FeatureExtractor
            self.WavLMModel = WavLMModel
            self.available = True
        except ImportError:
            warnings.warn(
                "Transformers 不可用。安装命令: pip install transformers"
            )
    
    def load_model(self, model_name: str = "microsoft/wavlm-base-plus"):
        """
        加载 WavLM 模型。
        
        Args:
            model_name: HuggingFace 模型名称
        """
        if not self.available:
            raise RuntimeError("Transformers 不可用")
        
        print(f"📥 加载 WavLM 模型: {model_name}")
        
        self.processor = self.FeatureExtractor.from_pretrained(model_name)
        self.model = self.WavLMModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ WavLM 模型加载完成 (设备: {self.device})")
    
    def _pad_audio_if_needed(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        如果音频过短，进行填充。
        
        Args:
            audio_segment: 输入音频采样点
        
        Returns:
            填充后的音频
        """
        if len(audio_segment) < self.MIN_AUDIO_LENGTH:
            padding_length = self.MIN_AUDIO_LENGTH - len(audio_segment)
            return np.pad(audio_segment, (0, padding_length), mode='constant')
        return audio_segment
    
    def extract_embedding(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        提取 WavLM 嵌入向量。
        
        Args:
            audio_segment: 音频采样点（numpy 数组）
        
        Returns:
            嵌入向量（numpy 数组）
        """
        if self.model is None:
            raise RuntimeError("模型未加载。请先调用 load_model()")
        
        if len(audio_segment) == 0:
            return np.zeros(self.EMBEDDING_DIM)
        
        original_length = len(audio_segment)
        audio_segment = self._pad_audio_if_needed(audio_segment)
        
        if original_length < self.MIN_AUDIO_LENGTH / 2:
            warnings.warn(
                f"音频片段过短 ({original_length} 采样点)，声学评分可能不准确。"
            )
        
        try:
            inputs = self.processor(
                audio_segment,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                # 时间维度平均池化
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embedding = embeddings.cpu().numpy()[0]
            return embedding
            
        except Exception as e:
            warnings.warn(f"嵌入提取失败: {e}")
            return np.zeros(self.EMBEDDING_DIM)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个嵌入向量的余弦相似度。
        
        Args:
            emb1: 第一个嵌入
            emb2: 第二个嵌入
        
        Returns:
            相似度得分 (0.0 到 1.0)
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.5  # 无效嵌入返回中性分数
        
        emb1_norm = emb1 / norm1
        emb2_norm = emb2 / norm2
        
        # 余弦相似度范围 [-1, 1]
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # 转换到 [0, 1] 范围
        similarity = (similarity + 1.0) / 2.0
        
        return float(similarity)
    
    def score_segments(
        self,
        sliced_results: List[Dict],
        reference_audio_path: Optional[str] = None,
        reference_segments: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        对发音片段进行声学评分。
        
        Args:
            sliced_results: AudioSlicer 输出的切片结果
            reference_audio_path: 可选的标准音路径（旧接口，保留兼容性）
            reference_segments: 可选的标准音切片列表（新接口，优先使用）
        
        Returns:
            添加声学得分的结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载。请先调用 load_model()")
        
        scored_results = []
        
        # 优先使用已切片的标准音
        reference_embeddings = None
        print(f"   🔍 检查 reference_segments: {reference_segments is not None}, 长度: {len(reference_segments) if reference_segments else 0}")
        
        if reference_segments and len(reference_segments) > 0:
            # 直接从切片提取嵌入
            print(f"   ✅ 使用标准音切片提取嵌入...")
            reference_embeddings = []
            for idx, seg in enumerate(reference_segments):
                ref_audio = seg.get("audio_segment", np.array([]))
                print(f"      [{idx}] {seg.get('char', '?')}: 音频长度={len(ref_audio)}")
                ref_embedding = self.extract_embedding(ref_audio)
                reference_embeddings.append(ref_embedding)
            print(f"   ✅ 提取了 {len(reference_embeddings)} 个标准音嵌入")
        elif reference_audio_path:
            # 兼容旧接口：从文件切片（不推荐）
            print(f"   ⚠️ 使用旧接口从文件切片")
            reference_embeddings = self._extract_reference_embeddings(
                reference_audio_path,
                sliced_results
            )
        else:
            print(f"   ⚠️ 无标准音，使用启发式评分")
        
        for i, item in enumerate(sliced_results):
            audio_segment = item.get("audio_segment", np.array([]))
            audio_length = len(audio_segment)
            
            # 提取用户嵌入
            user_embedding = self.extract_embedding(audio_segment)
            
            # 计算得分
            if reference_embeddings and i < len(reference_embeddings):
                # 有标准音：使用嵌入相似度
                ref_embedding = reference_embeddings[i]
                similarity = self.cosine_similarity(user_embedding, ref_embedding)
                
                # 调试日志
                char_name = item.get('char', f'[{i}]')
                print(f"   {char_name}: 余弦相似度={similarity:.4f}", end="")
                
                # 相似度映射到评分（提高对好发音的区分度）
                acoustic_score = self._similarity_to_score(similarity)
                print(f" → 声学得分={acoustic_score:.4f}")
            else:
                # 无标准音：使用启发式评分
                acoustic_score = self._heuristic_score(user_embedding, audio_segment)
            
            # 短音频惩罚
            if audio_length < self.MIN_AUDIO_LENGTH:
                length_ratio = audio_length / self.MIN_AUDIO_LENGTH
                acoustic_score *= max(0.5, length_ratio)
            
            # 添加结果
            result = item.copy()
            result["acoustic_score"] = float(acoustic_score)
            result["audio_length"] = audio_length
            scored_results.append(result)
        
        return scored_results
    
    def _similarity_to_score(self, similarity: float) -> float:
        """
        将余弦相似度映射到评分。
        
        WavLM 嵌入的余弦相似度特点：
        - 相同音频: ~0.99-1.0
        - 相似发音: ~0.85-0.95
        - 不同发音: ~0.6-0.8
        - 完全不同: <0.6
        
        Args:
            similarity: 余弦相似度 (0-1)
        
        Returns:
            评分 (0-1)
        """
        # 更宽容的映射：
        # similarity >= 0.9 -> score >= 0.95 (优秀)
        # similarity >= 0.8 -> score >= 0.85 (良好)
        # similarity >= 0.7 -> score >= 0.70 (及格)
        # similarity < 0.7 -> score < 0.70 (需改进)
        
        if similarity >= 0.9:
            # 优秀区间：0.9-1.0 -> 0.95-1.0
            score = 0.95 + (similarity - 0.9) * 0.5
        elif similarity >= 0.8:
            # 良好区间：0.8-0.9 -> 0.85-0.95
            score = 0.85 + (similarity - 0.8) * 1.0
        elif similarity >= 0.7:
            # 及格区间：0.7-0.8 -> 0.70-0.85
            score = 0.70 + (similarity - 0.7) * 1.5
        elif similarity >= 0.6:
            # 待改进：0.6-0.7 -> 0.55-0.70
            score = 0.55 + (similarity - 0.6) * 1.5
        else:
            # 较差：0-0.6 -> 0.3-0.55
            score = 0.3 + similarity * 0.42
        
        return min(1.0, max(0.0, score))
    
    def _extract_reference_embeddings(
        self,
        reference_audio_path: str,
        sliced_results: List[Dict]
    ) -> List[np.ndarray]:
        """
        从标准音中提取对应片段的嵌入。
        
        Args:
            reference_audio_path: 标准音路径
            sliced_results: 包含时间戳的切片结果
        
        Returns:
            参考嵌入列表
        """
        try:
            import librosa
        except ImportError:
            raise RuntimeError("librosa 不可用")
        
        try:
            ref_audio, _ = librosa.load(reference_audio_path, sr=16000, mono=True)
        except Exception as e:
            warnings.warn(f"加载标准音失败: {e}")
            return []
        
        ref_embeddings = []
        
        for item in sliced_results:
            start_time = item.get("start", 0)
            end_time = item.get("end", 0)
            
            start_sample = int(start_time * 16000)
            end_sample = int(end_time * 16000)
            
            start_sample = max(0, min(start_sample, len(ref_audio)))
            end_sample = max(0, min(end_sample, len(ref_audio)))
            
            if end_sample > start_sample:
                ref_segment = ref_audio[start_sample:end_sample]
            else:
                ref_segment = np.array([])
            
            ref_embedding = self.extract_embedding(ref_segment)
            ref_embeddings.append(ref_embedding)
        
        return ref_embeddings
    
    def _heuristic_score(
        self, 
        embedding: np.ndarray, 
        audio_segment: np.ndarray
    ) -> float:
        """
        无标准音时的启发式评分。
        
        基于嵌入统计量和音频特征。
        
        Args:
            embedding: 音频嵌入
            audio_segment: 原始音频
        
        Returns:
            启发式评分 (0-1)
        """
        scores = []
        
        # 1. 嵌入向量的有效性
        magnitude = np.linalg.norm(embedding)
        variance = np.var(embedding)
        
        # 有效嵌入应该有一定的量级和方差
        # 经验值：正常语音嵌入的 magnitude 约 15-25，variance 约 0.05-0.15
        mag_score = min(1.0, magnitude / 20.0)
        var_score = min(1.0, variance / 0.1)
        
        scores.append(0.5 * mag_score + 0.3 * var_score)
        
        # 2. 音频能量
        if len(audio_segment) > 0:
            rms = np.sqrt(np.mean(audio_segment ** 2))
            # 正常语音 RMS 约 0.01-0.1
            energy_score = min(1.0, rms * 20)
            scores.append(energy_score * 0.2)
        
        # 综合得分
        score = sum(scores)
        
        # 无标准音时，给予较保守的评分（避免虚高）
        # 映射到 0.5-0.85 范围
        score = 0.5 + score * 0.35
        
        return min(1.0, max(0.0, score))
    
    def is_available(self) -> bool:
        """检查 WavLM 是否可用。"""
        return self.available