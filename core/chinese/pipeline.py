"""
Chinese Pronunciation Scoring Pipeline - Enhanced Version

整合所有 9 个任务的完整发音评分流水线。
增强功能：
- 自动生成标准音（如果 TTS 可用）
- 使用 FunASR 进行中文识别（替代 WhisperX）
- 使用 F0 曲线分析进行声调评分（替代随机 MLP）
- 使用规则方法进行错误分类（替代随机 MLP）
"""

from typing import Dict, List, Optional
from pathlib import Path
import warnings
import logging
import tempfile
import os

from .pinyin_mapper import PinyinMapper
from .audio_aligner import ChineseAudioAligner
from .audio_slicer import AudioSlicer
from .acoustic_scorer import AcousticScorer
from .tone_scorer import ToneScorer
from .duration_scorer import DurationScorer
from .pause_scorer import PauseScorer
from .error_classifier import ErrorClassifier
from .final_scorer import FinalScorer

# 配置日志
logger = logging.getLogger(__name__)


class ChineseScoringPipeline:
    """
    完整的中文发音评分流水线。
    
    执行 9 个任务：
    1. 拼音映射 (PinyinMapper)
    2. 音频对齐 (ChineseAudioAligner) - FunASR
    3. 音频切片 (AudioSlicer)
    4. 声学评分 (AcousticScorer) - WavLM
    5. 声调评分 (ToneScorer) - F0 曲线分析
    6. 时长评分 (DurationScorer)
    7. 停顿评分 (PauseScorer)
    8. 错误分类 (ErrorClassifier) - 规则方法
    9. 综合评分 (FinalScorer)
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        enable_gpu: bool = True,
        tts_generator: Optional[callable] = None,
        reference_audio_dir: Optional[str] = None
    ):
        """
        初始化流水线。
        
        Args:
            device: 使用的设备 ('cuda' 或 'cpu')，None 时自动检测
            enable_gpu: 是否启用 GPU 加速
            tts_generator: TTS 生成函数，用于自动生成标准音
                          签名: tts_generator(text: str, output_path: str) -> bool
            reference_audio_dir: 预录制标准音的目录（可选）
        """
        # 设备检测
        if device is None and enable_gpu:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        elif device is None:
            device = "cpu"
        
        self.device = device
        self.tts_generator = tts_generator
        self.reference_audio_dir = Path(reference_audio_dir) if reference_audio_dir else None
        
        # 初始化所有模块
        self.pinyin_mapper = PinyinMapper()
        self.audio_aligner = ChineseAudioAligner(device=device)
        self.audio_slicer = AudioSlicer()
        self.acoustic_scorer = AcousticScorer(device=device)
        self.tone_scorer = ToneScorer(device=device)
        self.duration_scorer = DurationScorer()
        self.pause_scorer = PauseScorer()
        self.error_classifier = ErrorClassifier(device=device)
        self.final_scorer = FinalScorer()
        
        # 标准音缓存（避免重复生成）
        self._reference_cache: Dict[str, str] = {}
        
        # 临时文件目录
        self._temp_dir = Path(tempfile.gettempdir()) / "chinese_scoring"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_loaded = False
    
    def set_tts_generator(self, tts_generator: callable):
        """
        设置 TTS 生成器。
        
        Args:
            tts_generator: TTS 生成函数
                          签名: tts_generator(text: str, output_path: str) -> bool
        """
        self.tts_generator = tts_generator
        logger.info("TTS 生成器已设置")
    
    def load_models(self, model_size: str = "base"):
        """
        加载所有必需的模型。
        
        Args:
            model_size: 模型大小（用于 WhisperX 备选）
        """
        if self.models_loaded:
            return
        
        logger.info("正在加载中文发音评分模型...")
        
        # 1. 加载音频对齐模型 (FunASR)
        if self.audio_aligner.is_available():
            try:
                self.audio_aligner.load_models(model_size)
            except Exception as e:
                warnings.warn(f"音频对齐模型加载失败: {e}")
        
        # 2. 加载声学评分模型 (WavLM)
        if self.acoustic_scorer.is_available():
            try:
                self.acoustic_scorer.load_model()
            except Exception as e:
                warnings.warn(f"声学评分模型加载失败: {e}")
        
        # 3. 加载声调评分模块
        if self.tone_scorer.is_available():
            try:
                self.tone_scorer.load_models()
            except Exception as e:
                warnings.warn(f"声调评分模块加载失败: {e}")
        
        # 4. 加载错误分类模块
        if self.error_classifier.is_available():
            try:
                self.error_classifier.load_models()
            except Exception as e:
                warnings.warn(f"错误分类模块加载失败: {e}")
        
        self.models_loaded = True
        logger.info(f"模型加载完成 (设备: {self.device})")
    
    def _get_or_generate_reference_audio(
        self,
        reference_text: str,
        force_generate: bool = False
    ) -> Optional[str]:
        """
        获取或生成标准音音频。
        
        优先级：
        1. 缓存中的标准音
        2. 预录制的标准音文件
        3. TTS 生成的标准音
        
        Args:
            reference_text: 参考文本
            force_generate: 是否强制重新生成
        
        Returns:
            标准音音频路径，或 None 如果不可用
        """
        # 生成缓存键
        cache_key = reference_text.strip()
        
        # 1. 检查缓存
        if not force_generate and cache_key in self._reference_cache:
            cached_path = self._reference_cache[cache_key]
            if Path(cached_path).exists():
                logger.debug(f"使用缓存的标准音: {cached_path}")
                return cached_path
        
        # 2. 检查预录制的标准音
        if self.reference_audio_dir and self.reference_audio_dir.exists():
            # 尝试多种命名方式
            possible_names = [
                f"{cache_key}.wav",
                f"{cache_key.replace(' ', '_')}.wav",
                f"ref_{hash(cache_key) % 10000}.wav"
            ]
            
            for name in possible_names:
                ref_path = self.reference_audio_dir / name
                if ref_path.exists():
                    self._reference_cache[cache_key] = str(ref_path)
                    logger.debug(f"使用预录制的标准音: {ref_path}")
                    return str(ref_path)
        
        # 3. 使用 TTS 生成
        if self.tts_generator is not None:
            try:
                # 生成输出路径
                safe_filename = "".join(c if c.isalnum() else "_" for c in cache_key[:20])
                output_path = self._temp_dir / f"ref_{safe_filename}_{hash(cache_key) % 10000}.wav"
                
                logger.info(f"正在生成标准音: {reference_text[:20]}...")
                
                success = self.tts_generator(reference_text, str(output_path))
                
                if success and output_path.exists():
                    self._reference_cache[cache_key] = str(output_path)
                    logger.info(f"标准音生成成功: {output_path}")
                    return str(output_path)
                else:
                    logger.warning("TTS 生成失败")
            except Exception as e:
                logger.warning(f"TTS 生成异常: {e}")
        
        logger.debug("无可用的标准音")
        return None
    
    def score_pronunciation(
        self,
        audio_path: str,
        reference_text: str,
        reference_audio_path: Optional[str] = None,
        auto_generate_reference: bool = True
    ) -> Dict:
        """
        对中文发音进行完整评分。
        
        Args:
            audio_path: 用户录音文件路径
            reference_text: 期望的文本
            reference_audio_path: 标准音文件路径（可选）
            auto_generate_reference: 是否自动生成标准音（如果未提供）
        
        Returns:
            完整的评分结果字典
        """
        if not self.models_loaded:
            self.load_models()
        
        # 自动获取/生成标准音
        if reference_audio_path is None and auto_generate_reference:
            reference_audio_path = self._get_or_generate_reference_audio(reference_text)
        
        has_reference = reference_audio_path is not None and Path(reference_audio_path).exists()
        
        if has_reference:
            logger.info("使用标准音进行评分（高精度模式）")
        else:
            logger.info("无标准音，使用模式分析评分")
        
        # ============ Task 1: 拼音映射 ============
        logger.info("Task 1: 文本 → 拼音...")
        pinyin_sequence = self.pinyin_mapper.text_to_pinyin(reference_text)
        
        if not pinyin_sequence:
            raise ValueError("参考文本中没有找到中文字符")
        
        # ============ Task 2: 音频对齐 ============
        logger.info("Task 2: 音频对齐（FunASR）...")
        if self.audio_aligner.is_available():
            alignment_results = self.audio_aligner.align_audio(
                audio_path,
                pinyin_sequence
            )
        else:
            warnings.warn("音频对齐不可用，使用均匀分配")
            alignment_results = self._create_placeholder_alignment(
                audio_path,
                pinyin_sequence
            )
        
        # ============ Task 3: 音频切片 ============
        logger.info("Task 3: 音频切片...")
        sliced_results = self.audio_slicer.slice_audio(
            audio_path,
            alignment_results
        )
        
        # 调试：打印用户音频切片信息
        print(f"\n📊 用户音频切片结果 ({len(sliced_results)} 个):")
        for i, seg in enumerate(sliced_results[:5]):  # 只打印前5个
            audio_len = len(seg.get('audio_segment', []))
            print(f"   [{i}] {seg.get('char')}: {seg.get('start', 0):.3f}s-{seg.get('end', 0):.3f}s, 采样点={audio_len}")
        if len(sliced_results) > 5:
            print(f"   ... 共 {len(sliced_results)} 个")
        
        # 如果有标准音，对标准音进行**独立对齐和切片**
        reference_segments = None
        if has_reference:
            try:
                # 标准音需要独立对齐，不能用用户录音的时间戳！
                logger.info("对标准音进行独立对齐...")
                ref_alignment_results = self.audio_aligner.align_audio(
                    reference_audio_path,
                    pinyin_sequence
                )
                
                # 调试：打印标准音对齐结果
                print(f"\n📊 标准音对齐结果 ({len(ref_alignment_results)} 个):")
                for i, seg in enumerate(ref_alignment_results[:5]):
                    print(f"   [{i}] {seg.get('char')}: {seg.get('start', 0):.3f}s-{seg.get('end', 0):.3f}s")
                if len(ref_alignment_results) > 5:
                    print(f"   ... 共 {len(ref_alignment_results)} 个")
                
                # 用标准音自己的时间戳切片
                reference_segments = self.audio_slicer.slice_audio(
                    reference_audio_path,
                    ref_alignment_results
                )
                
                # 调试：打印标准音切片信息
                print(f"\n📊 标准音切片结果 ({len(reference_segments)} 个):")
                for i, seg in enumerate(reference_segments[:5]):
                    audio_len = len(seg.get('audio_segment', []))
                    print(f"   [{i}] {seg.get('char')}: 采样点={audio_len}")
                
                logger.info(f"标准音切片完成: {len(reference_segments)} 段")
            except Exception as e:
                import traceback
                warnings.warn(f"标准音切片失败: {e}")
                traceback.print_exc()
                reference_segments = None
        
        # ============ Task 4: 声学评分 ============
        logger.info("Task 4: 声学评分（WavLM）...")
        print(f"\n🔊 开始声学评分...")
        print(f"   reference_segments 是否存在: {reference_segments is not None}")
        if reference_segments:
            print(f"   reference_segments 数量: {len(reference_segments)}")
        
        if self.acoustic_scorer.is_available():
            sliced_results = self.acoustic_scorer.score_segments(
                sliced_results,
                reference_segments=reference_segments
            )
            
            # 打印声学评分结果
            print(f"\n📊 声学评分结果:")
            for i, item in enumerate(sliced_results[:5]):
                print(f"   {item.get('char')}: acoustic_score={item.get('acoustic_score', 'N/A'):.4f}")
        else:
            for item in sliced_results:
                item["acoustic_score"] = 0.7
        
        # ============ Task 5: 声调评分 ============
        logger.info("Task 5: 声调评分（F0 分析）...")
        print(f"\n🎵 开始声调评分...")
        
        if self.tone_scorer.is_available():
            sliced_results = self.tone_scorer.score_tones(
                sliced_results,
                reference_segments
            )
            
            # 打印声调评分结果
            print(f"\n📊 声调评分结果:")
            for i, item in enumerate(sliced_results[:5]):
                print(f"   {item.get('char')}: tone_score={item.get('tone_score', 'N/A'):.4f}, "
                      f"predicted={item.get('predicted_tone')}, expected={item.get('expected_tone')}")
        else:
            for item in sliced_results:
                item["tone_score"] = 0.7
                item["predicted_tone"] = 0
                item["expected_tone"] = 0
        
        # ============ Task 6: 时长评分 ============
        logger.info("Task 6: 时长评分...")
        reference_durations = None
        if reference_segments:
            reference_durations = [
                seg.get('end', 0) - seg.get('start', 0) 
                for seg in reference_segments
            ]
        sliced_results = self.duration_scorer.score_durations(
            sliced_results,
            reference_durations
        )
        
        # ============ Task 7: 停顿评分 ============
        logger.info("Task 7: 停顿/流畅度评分...")
        sliced_results = self.pause_scorer.score_pauses(sliced_results)
        fluency_metrics = self.pause_scorer.calculate_overall_fluency(sliced_results)
        
        # ============ Task 8: 错误分类 ============
        logger.info("Task 8: 错误分类...")
        if self.error_classifier.is_available():
            sliced_results = self.error_classifier.classify_errors(sliced_results)
        else:
            for item in sliced_results:
                item["errors"] = []
                item["error_probabilities"] = {}
        
        # ============ Task 9: 综合评分 ============
        logger.info("Task 9: 综合评分...")
        final_results = self.final_scorer.calculate_final_scores(sliced_results)
        overall_metrics = self.final_scorer.calculate_overall_score(final_results)
        feedback = self.final_scorer.generate_feedback(final_results, overall_metrics)
        
        # 打印各维度得分汇总
        print(f"\n" + "="*60)
        print(f"📊 各字符详细得分:")
        print(f"{'字符':<4} {'声学':<8} {'声调':<8} {'时长':<8} {'停顿':<8} {'最终':<8}")
        print("-"*60)
        for item in final_results:
            char = item.get('char', '?')
            acoustic = item.get('acoustic_score', 0) * 100
            tone = item.get('tone_score', 0) * 100
            duration = item.get('duration_score', 0) * 100
            pause = item.get('pause_score', 0) * 100
            final = item.get('final_score', 0)
            print(f"{char:<4} {acoustic:<8.1f} {tone:<8.1f} {duration:<8.1f} {pause:<8.1f} {final:<8}")
        print("="*60)
        print(f"总分: {overall_metrics['overall_score']}/100")
        print(f"="*60 + "\n")
        
        # 添加错误改进建议
        all_errors = []
        for item in final_results:
            all_errors.extend(item.get('errors', []))
        
        if all_errors:
            error_suggestions = self.error_classifier.get_error_suggestions(
                list(set(all_errors))
            )
            feedback.extend(error_suggestions)
        
        # 编译最终结果
        results = {
            "overall_score": overall_metrics["overall_score"],
            "character_scores": self._clean_results_for_output(final_results),
            "overall_metrics": overall_metrics,
            "fluency_metrics": fluency_metrics,
            "feedback": feedback,
            "reference_text": reference_text,
            "num_characters": len(final_results),
            "has_reference_audio": has_reference,
            "scoring_mode": "reference_comparison" if has_reference else "pattern_analysis"
        }
        
        logger.info(f"评分完成！总分: {results['overall_score']}/100")
        
        return results
    
    def _clean_results_for_output(self, results: List[Dict]) -> List[Dict]:
        """
        清理结果，移除不需要序列化的数据（如 numpy 数组）。
        
        Args:
            results: 原始结果列表
        
        Returns:
            清理后的结果列表
        """
        cleaned = []
        
        for item in results:
            cleaned_item = {}
            
            for key, value in item.items():
                # 跳过 numpy 数组
                if key == 'audio_segment':
                    continue
                
                # 跳过复杂的特征字典
                if key == 'f0_features':
                    continue
                
                # 转换 numpy 类型
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, 'tolist'):
                    value = value.tolist()
                
                cleaned_item[key] = value
            
            cleaned.append(cleaned_item)
        
        return cleaned
    
    def _create_placeholder_alignment(
        self,
        audio_path: str,
        pinyin_sequence: List[Dict]
    ) -> List[Dict]:
        """
        创建占位对齐结果（当 ASR 不可用时）。
        
        Args:
            audio_path: 音频文件路径
            pinyin_sequence: 拼音序列
        
        Returns:
            均匀分配的对齐结果
        """
        try:
            import librosa
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            total_duration = len(audio) / 16000
        except:
            total_duration = len(pinyin_sequence) * 0.25
        
        num_chars = len(pinyin_sequence)
        char_duration = total_duration / num_chars if num_chars > 0 else 0
        
        alignment_results = []
        for i, item in enumerate(pinyin_sequence):
            alignment_results.append({
                "char": item["char"],
                "pinyin": item["pinyin"],
                "start": i * char_duration,
                "end": (i + 1) * char_duration,
                "score": 0.5
            })
        
        return alignment_results
    
    def clear_cache(self):
        """清除标准音缓存。"""
        self._reference_cache.clear()
        
        # 清理临时文件
        if self._temp_dir.exists():
            for file in self._temp_dir.glob("ref_*.wav"):
                try:
                    file.unlink()
                except:
                    pass
        
        logger.info("缓存已清除")
    
    def is_available(self) -> bool:
        """
        检查流水线是否可用。
        
        Returns:
            True 如果基本功能可用
        """
        return self.pinyin_mapper.is_available()