"""
Task 8: Error Classification - Rule-based Version

基于规则的发音错误分类器。
不再使用随机初始化的 MLP，而是根据各维度评分综合判断。

错误类型：
- 声母轻 (initial_weak): 声母发音不清晰
- 韵母不圆 (final_not_round): 韵母发音不到位
- 声调错误 (tone_error): 声调不准确
- 发音模糊 (unclear): 整体发音模糊
- 时长异常 (duration_abnormal): 发音过短或过长
- 停顿过多 (excessive_pause): 不流畅
"""

from typing import List, Dict, Optional
import numpy as np
import warnings
import re


class ErrorClassifier:
    """
    基于规则的发音错误分类器。
    
    根据各维度评分（声学、声调、时长、停顿）综合判断错误类型。
    """
    
    # 错误类型定义
    ERROR_TYPES = {
        'tone_error': {
            'zh': '声调错误',
            'en': 'Wrong tone',
            'description': '声调与标准发音不一致'
        },
        'unclear': {
            'zh': '发音模糊',
            'en': 'Unclear pronunciation',
            'description': '整体发音不清晰'
        },
        'initial_weak': {
            'zh': '声母轻',
            'en': 'Weak initial consonant',
            'description': '声母发音不够有力'
        },
        'final_not_round': {
            'zh': '韵母不圆',
            'en': 'Final vowel not round',
            'description': '韵母发音不到位'
        },
        'duration_short': {
            'zh': '发音过短',
            'en': 'Too short',
            'description': '发音时间过短'
        },
        'duration_long': {
            'zh': '发音过长',
            'en': 'Too long',
            'description': '发音时间过长，拖音'
        },
        'pause_excessive': {
            'zh': '停顿过多',
            'en': 'Excessive pauses',
            'description': '字间停顿过长，不流畅'
        }
    }
    
    # 评分阈值
    THRESHOLDS = {
        'tone_error': 0.6,         # 声调得分低于此值判定为声调错误
        'unclear': 0.5,            # 声学得分低于此值判定为发音模糊
        'initial_weak': 0.55,      # 声学得分阈值（配合拼音判断）
        'final_not_round': 0.55,   # 声学得分阈值（配合拼音判断）
        'duration_short': 0.08,    # 时长低于此值（秒）判定为过短
        'duration_long': 0.8,      # 时长高于此值（秒）判定为过长
        'pause_excessive': 0.4,    # 停顿得分低于此值判定为停顿过多
    }
    
    # 需要圆唇的韵母（用于判断"韵母不圆"）
    ROUND_FINALS = ['u', 'o', 'ü', 'ou', 'uo', 'üe', 'un', 'ong', 'iong']
    
    # 需要有力发音的声母（用于判断"声母轻"）
    STRONG_INITIALS = ['b', 'p', 'd', 't', 'g', 'k', 'zh', 'ch', 'z', 'c']
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化错误分类器。
        
        Args:
            device: 设备（保留参数兼容性，此版本不需要 GPU）
        """
        self.device = device
        self.available = True
    
    def load_models(self, model_name: str = None):
        """
        加载模型（兼容旧接口）。
        
        此版本使用规则方法，不需要加载神经网络模型。
        """
        print("✅ 错误分类器初始化完成（规则分析模式）")
    
    def classify_errors(
        self,
        sliced_results: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        对每个字符的发音进行错误分类。
        
        Args:
            sliced_results: 包含各维度评分的切片结果
            threshold: 错误判定的置信度阈值（保留兼容性）
        
        Returns:
            添加错误分类的结果列表
        """
        classified_results = []
        
        for i, item in enumerate(sliced_results):
            # 提取各维度得分
            acoustic_score = item.get('acoustic_score', 0.7)
            tone_score = item.get('tone_score', 0.7)
            duration_score = item.get('duration_score', 0.7)
            pause_score = item.get('pause_score', 0.7)
            duration = item.get('duration', 0.25)
            pinyin = item.get('pinyin', '')
            
            # 分类错误
            errors = []
            error_probabilities = {}
            
            # 1. 声调错误检测
            tone_error_prob = self._detect_tone_error(item)
            error_probabilities['声调错误'] = tone_error_prob
            if tone_error_prob > threshold:
                errors.append('声调错误')
            
            # 2. 发音模糊检测
            unclear_prob = self._detect_unclear(item)
            error_probabilities['发音模糊'] = unclear_prob
            if unclear_prob > threshold:
                errors.append('发音模糊')
            
            # 3. 声母轻检测
            initial_weak_prob = self._detect_initial_weak(item, pinyin)
            error_probabilities['声母轻'] = initial_weak_prob
            if initial_weak_prob > threshold:
                errors.append('声母轻')
            
            # 4. 韵母不圆检测
            final_not_round_prob = self._detect_final_not_round(item, pinyin)
            error_probabilities['韵母不圆'] = final_not_round_prob
            if final_not_round_prob > threshold:
                errors.append('韵母不圆')
            
            # 5. 时长异常检测
            duration_error, duration_error_prob = self._detect_duration_error(duration)
            if duration_error:
                error_probabilities[duration_error] = duration_error_prob
                if duration_error_prob > threshold:
                    errors.append(duration_error)
            
            # 6. 停顿过多检测（基于 pause_score）
            pause_error_prob = self._detect_pause_error(pause_score)
            error_probabilities['停顿过多'] = pause_error_prob
            if pause_error_prob > threshold:
                errors.append('停顿过多')
            
            # 去重并排序（按概率从高到低）
            errors = list(set(errors))
            errors.sort(key=lambda x: error_probabilities.get(x, 0), reverse=True)
            
            # 添加结果
            result = item.copy()
            result['errors'] = errors
            result['error_probabilities'] = error_probabilities
            
            classified_results.append(result)
        
        return classified_results
    
    def _detect_tone_error(self, item: Dict) -> float:
        """
        检测声调错误。
        
        Args:
            item: 包含评分信息的字典
        
        Returns:
            错误概率 (0-1)
        """
        tone_score = item.get('tone_score', 0.7)
        predicted_tone = item.get('predicted_tone', 0)
        expected_tone = item.get('expected_tone', 0)
        
        # 基础概率：1 - tone_score
        base_prob = 1.0 - tone_score
        
        # 如果预测声调与期望不符，增加概率
        if predicted_tone != expected_tone and predicted_tone > 0 and expected_tone > 0:
            base_prob = max(base_prob, 0.6)
        
        return min(1.0, base_prob)
    
    def _detect_unclear(self, item: Dict) -> float:
        """
        检测发音模糊。
        
        Args:
            item: 包含评分信息的字典
        
        Returns:
            错误概率 (0-1)
        """
        acoustic_score = item.get('acoustic_score', 0.7)
        duration = item.get('duration', 0.25)
        
        # 基础概率：1 - acoustic_score
        base_prob = 1.0 - acoustic_score
        
        # 时长过短增加模糊概率
        if duration < 0.1:
            base_prob += 0.2
        
        return min(1.0, base_prob)
    
    def _detect_initial_weak(self, item: Dict, pinyin: str) -> float:
        """
        检测声母发音过轻。
        
        只对需要有力发音的声母进行检测。
        
        Args:
            item: 包含评分信息的字典
            pinyin: 拼音
        
        Returns:
            错误概率 (0-1)
        """
        acoustic_score = item.get('acoustic_score', 0.7)
        
        # 获取声母
        initial = self._get_initial(pinyin)
        
        # 只有特定声母才检测
        if initial not in self.STRONG_INITIALS:
            return 0.0
        
        # 声学得分低可能表示声母不清晰
        if acoustic_score < self.THRESHOLDS['initial_weak']:
            return (self.THRESHOLDS['initial_weak'] - acoustic_score) * 2
        
        return 0.0
    
    def _detect_final_not_round(self, item: Dict, pinyin: str) -> float:
        """
        检测韵母不圆。
        
        只对圆唇韵母进行检测。
        
        Args:
            item: 包含评分信息的字典
            pinyin: 拼音
        
        Returns:
            错误概率 (0-1)
        """
        acoustic_score = item.get('acoustic_score', 0.7)
        
        # 获取韵母
        final = self._get_final(pinyin)
        
        # 检查是否是圆唇韵母
        is_round_final = any(rf in final for rf in self.ROUND_FINALS)
        
        if not is_round_final:
            return 0.0
        
        # 声学得分低可能表示韵母不到位
        if acoustic_score < self.THRESHOLDS['final_not_round']:
            return (self.THRESHOLDS['final_not_round'] - acoustic_score) * 2
        
        return 0.0
    
    def _detect_duration_error(self, duration: float) -> tuple:
        """
        检测时长异常。
        
        Args:
            duration: 发音时长（秒）
        
        Returns:
            (错误类型, 错误概率) 或 (None, 0)
        """
        if duration < self.THRESHOLDS['duration_short']:
            # 发音过短
            prob = (self.THRESHOLDS['duration_short'] - duration) / self.THRESHOLDS['duration_short']
            return ('发音过短', min(1.0, prob))
        
        elif duration > self.THRESHOLDS['duration_long']:
            # 发音过长
            excess = duration - self.THRESHOLDS['duration_long']
            prob = min(1.0, excess / 0.5)
            return ('发音过长', prob)
        
        return (None, 0.0)
    
    def _detect_pause_error(self, pause_score: float) -> float:
        """
        检测停顿过多。
        
        Args:
            pause_score: 停顿/流畅度得分
        
        Returns:
            错误概率 (0-1)
        """
        if pause_score < self.THRESHOLDS['pause_excessive']:
            return (self.THRESHOLDS['pause_excessive'] - pause_score) * 2
        
        return 0.0
    
    def _get_initial(self, pinyin: str) -> str:
        """
        从拼音中提取声母。
        
        Args:
            pinyin: 带声调的拼音（如 "ni3"）
        
        Returns:
            声母
        """
        # 去除声调数字
        base_pinyin = re.sub(r'\d$', '', pinyin.lower())
        
        # 声母列表（按长度降序，先匹配长的）
        initials = [
            'zh', 'ch', 'sh',  # 双字母声母
            'b', 'p', 'm', 'f',
            'd', 't', 'n', 'l',
            'g', 'k', 'h',
            'j', 'q', 'x',
            'r', 'z', 'c', 's',
            'y', 'w'
        ]
        
        for initial in initials:
            if base_pinyin.startswith(initial):
                return initial
        
        return ''  # 零声母
    
    def _get_final(self, pinyin: str) -> str:
        """
        从拼音中提取韵母。
        
        Args:
            pinyin: 带声调的拼音（如 "ni3"）
        
        Returns:
            韵母
        """
        # 去除声调数字
        base_pinyin = re.sub(r'\d$', '', pinyin.lower())
        
        # 去除声母，剩下的就是韵母
        initial = self._get_initial(pinyin)
        
        if initial:
            return base_pinyin[len(initial):]
        
        return base_pinyin  # 零声母，整个都是韵母
    
    def get_error_suggestions(self, errors: List[str]) -> List[str]:
        """
        根据错误类型生成改进建议。
        
        Args:
            errors: 错误类型列表
        
        Returns:
            改进建议列表
        """
        suggestions = []
        
        error_suggestions = {
            '声调错误': '注意声调的高低变化，一声高平、二声上升、三声先降后升、四声下降。',
            '发音模糊': '发音时嘴型要到位，吐字要清晰。',
            '声母轻': '声母发音要有力，注意气流的控制。',
            '韵母不圆': '韵母发音时嘴型要圆，特别是 u、o 等圆唇元音。',
            '发音过短': '发音时间不要太短，每个字要发完整。',
            '发音过长': '避免拖音，保持自然的发音节奏。',
            '停顿过多': '字与字之间要连贯，减少不必要的停顿。'
        }
        
        for error in errors:
            if error in error_suggestions:
                suggestions.append(error_suggestions[error])
        
        return suggestions
    
    def is_available(self) -> bool:
        """检查分类器是否可用。"""
        return self.available