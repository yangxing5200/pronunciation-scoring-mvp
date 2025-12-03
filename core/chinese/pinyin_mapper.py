"""
Task 1: Pinyin Mapping (Text → Phoneme)

Converts Chinese text to pinyin with tone marks.
Uses pypinyin for offline processing.
"""

from typing import List, Dict, Optional
import re


class PinyinMapper:
    """
    Maps Chinese text to pinyin with tone numbers.
    
    Uses pypinyin library for offline, accurate pinyin conversion.
    """
    
    def __init__(self):
        """Initialize pinyin mapper."""
        try:
            from pypinyin import pinyin, Style
            self.pinyin = pinyin
            self.Style = Style
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: pypinyin not installed. Install with: pip install pypinyin")
    
    def text_to_pinyin(self, text: str) -> List[Dict[str, str]]:
        """
        Convert Chinese text to pinyin with tone numbers.
        
        Args:
            text: Chinese text string
        
        Returns:
            List of dictionaries with 'char' and 'pinyin' keys
            Example: [{"char":"你", "pinyin":"ni3"}, {"char":"好", "pinyin":"hao3"}]
        """
        if not self.available:
            raise RuntimeError("pypinyin not available. Please install: pip install pypinyin")
        
        # Extract only Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        if not chinese_chars:
            return []
        
        # Convert to pinyin with tone numbers (Style.TONE3)
        # heteronym=False to get the most common pronunciation
        pinyin_list = self.pinyin(chinese_chars, style=self.Style.TONE3, heteronym=False)
        
        result = []
        for i, char in enumerate(chinese_chars):
            # pinyin_list[i] is a list with one element (since heteronym=False)
            char_pinyin = pinyin_list[i][0] if pinyin_list[i] else ""
            result.append({
                "char": char,
                "pinyin": char_pinyin
            })
        
        return result
    
    def text_to_pinyin_with_alternatives(self, text: str) -> List[Dict[str, any]]:
        """
        Convert Chinese text to pinyin with alternative pronunciations.
        
        Args:
            text: Chinese text string
        
        Returns:
            List of dictionaries with 'char' and 'pinyin' (list) keys
            Example: [{"char":"好", "pinyin":["hao3", "hao4"]}]
        """
        if not self.available:
            raise RuntimeError("pypinyin not available. Please install: pip install pypinyin")
        
        # Extract only Chinese characters
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        
        if not chinese_chars:
            return []
        
        # Convert to pinyin with tone numbers, including alternatives
        pinyin_list = self.pinyin(chinese_chars, style=self.Style.TONE3, heteronym=True)
        
        result = []
        for i, char in enumerate(chinese_chars):
            result.append({
                "char": char,
                "pinyin": pinyin_list[i]  # List of alternative pronunciations
            })
        
        return result
    
    def get_tone(self, pinyin_str: str) -> int:
        """
        Extract tone number from pinyin string.
        
        Args:
            pinyin_str: Pinyin string with tone number (e.g., "ni3", "hao3")
        
        Returns:
            Tone number (1-5, where 5 is neutral tone) or 0 if not found
        """
        # Extract last digit if present
        match = re.search(r'(\d)$', pinyin_str)
        if match:
            return int(match.group(1))
        return 0
    
    def get_initial_final(self, pinyin_str: str) -> Dict[str, str]:
        """
        Split pinyin into initial (consonant) and final (vowel) parts.
        
        Args:
            pinyin_str: Pinyin string with tone (e.g., "ni3", "hao3")
        
        Returns:
            Dictionary with 'initial' and 'final' keys
        """
        # Remove tone number
        base_pinyin = re.sub(r'\d$', '', pinyin_str)
        
        # Common Chinese initials (声母)
        initials = [
            'b', 'p', 'm', 'f',
            'd', 't', 'n', 'l',
            'g', 'k', 'h',
            'j', 'q', 'x',
            'zh', 'ch', 'sh', 'r',
            'z', 'c', 's',
            'y', 'w'
        ]
        
        # Find matching initial
        initial = ""
        final = base_pinyin
        
        # Sort by length (descending) to match 'zh', 'ch', 'sh' before single letters
        for init in sorted(initials, key=len, reverse=True):
            if base_pinyin.startswith(init):
                initial = init
                final = base_pinyin[len(init):]
                break
        
        return {
            "initial": initial,
            "final": final
        }
    
    def is_available(self) -> bool:
        """Check if pypinyin is available."""
        return self.available
