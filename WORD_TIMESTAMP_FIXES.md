# Word-by-Word Playback and Chinese Character Segmentation Fixes

## Overview

This document describes the fixes implemented to address three main issues:
1. **English word playback timing inaccuracy** - Words were being played at wrong timestamps
2. **Chinese text not split into individual characters** - Chinese text was treated as whole words
3. **JavaScript playback precision issues** - setTimeout was not precise enough

## Changes Made

### 1. Core/Transcriber.py

**Added Methods:**

- `_is_chinese(text: str) -> bool`: Detects if text contains Chinese characters (U+4E00 to U+9FFF)
- `_split_chinese_characters(text, start_time, end_time) -> List[Dict]`: Splits Chinese text into individual characters with proportionally allocated timestamps

**Modified Methods:**

- `transcribe()`: Now detects Chinese text and automatically splits it into individual characters with proper timing

**Implementation Details:**

```python
# Chinese character detection
def _is_chinese(self, text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# Chinese character splitting with time allocation
def _split_chinese_characters(self, text, start_time, end_time):
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    total_duration = end_time - start_time
    char_duration = total_duration / len(chinese_chars)
    
    # Allocate proportional time to each character
    result = []
    for i, char in enumerate(chinese_chars):
        result.append({
            'word': char,
            'start': start_time + i * char_duration,
            'end': start_time + (i + 1) * char_duration,
            'probability': 1.0
        })
    return result
```

### 2. Core/Scorer.py

**Added Methods:**

- `_is_chinese(text: str) -> bool`: Same Chinese detection as in transcriber
- `_split_chinese_text(text: str) -> List[str]`: Extracts Chinese characters from text

**Modified Methods:**

- `_remove_punctuation()`: Now removes both English and Chinese punctuation marks (，。！？；：""''（）《》【】、)
- `_score_words()`: Automatically detects Chinese text and splits into characters before scoring

**Implementation Details:**

```python
# Updated word scoring to handle Chinese
if self._is_chinese(ref_clean):
    # For Chinese, split into characters
    ref_words = self._split_chinese_text(ref_clean)
    trans_words = self._split_chinese_text(trans_clean)
else:
    # For non-Chinese, split by whitespace
    ref_words = ref_clean.split()
    trans_words = trans_clean.split()
```

### 3. App.py

**Added Functions:**

- `find_word_timestamp(word, word_timestamps)`: Finds timestamp by matching word text instead of relying on index position

**Modified Sections:**

1. **Word Timestamp Matching**: Changed from index-based to text-based matching
   ```python
   # OLD: Using index
   if idx < len(word_timestamps):
       start_time = word_timestamps[idx].get('start', 0)
   
   # NEW: Using text matching
   word_ts = find_word_timestamp(word, word_timestamps)
   if word_ts:
       start_time = word_ts.get('start', 0)
   ```

2. **JavaScript Playback**: Replaced setTimeout with requestAnimationFrame for precise timing
   ```javascript
   // OLD: setTimeout based
   const duration = (endTime - startTime) * 1000;
   setTimeout(() => { audio.pause(); }, duration);
   
   // NEW: requestAnimationFrame based
   function checkTime() {
       if (audio.currentTime >= endTime - 0.01) {
           audio.pause();
       } else if (!audio.paused) {
           requestAnimationFrame(checkTime);
       }
   }
   audio.play().then(() => {
       requestAnimationFrame(checkTime);
   });
   ```

## Technical Rationale

### Why Text-Based Matching?

The root cause of timing issues was that `word_scores` is based on the reference text, while `word_timestamps` is based on Whisper transcription. When transcription differs from reference (e.g., missing words), index-based matching fails.

**Example:**
- Reference: ["hello", "world", "this", "is", "a", "test"]
- Transcribed: ["hello", "world", "this", "is", "test"] (missing "a")
- Click on "test" (index 5) → plays word_timestamps[5] which might be wrong

Text-based matching solves this by finding the actual word in the timestamp list.

### Why Character Splitting for Chinese?

Whisper returns Chinese text as phrases or whole sentences, not individual characters. For pronunciation practice, users need to hear each character separately.

**Time Allocation:**
- Total duration is divided equally among all Chinese characters
- This is a reasonable approximation since Chinese characters typically have similar duration
- Formula: `char_duration = total_duration / num_characters`

### Why requestAnimationFrame?

`setTimeout` is not precise enough for audio control because:
- It's not synchronized with the browser's refresh cycle
- Minimum delay varies by browser (4-10ms)
- Can drift over time

`requestAnimationFrame`:
- Runs at ~60fps (16.67ms intervals)
- Synchronized with browser rendering
- More accurate for real-time control
- The 0.01s (10ms) offset prevents playing into the next word

## Testing

Created comprehensive test suite in `scripts/test_word_timestamp_fixes.py`:

1. **Chinese Detection Tests**: Validates detection of Chinese vs non-Chinese text
2. **Chinese Splitting Tests**: Verifies character extraction and time allocation
3. **Word Matching Tests**: Tests exact and fuzzy matching logic
4. **Scorer Integration Tests**: Validates Chinese support in scoring
5. **Transcriber Integration Tests**: Confirms transcriber handles Chinese properly

All tests pass successfully.

## Backward Compatibility

These changes are fully backward compatible:
- English text continues to work as before
- Existing word scoring logic unchanged for non-Chinese text
- No breaking changes to APIs or data structures

## Future Improvements

1. **Language-Specific Time Allocation**: Different languages may need different time allocation strategies
2. **Mixed Language Support**: Better handling of text with both Chinese and English
3. **Pinyin Support**: Add pinyin display for Chinese characters
4. **Tone Analysis**: Evaluate tone accuracy for Chinese pronunciation
5. **Alternative Time Estimation**: Use acoustic features instead of equal division

## Related Issues

This implementation addresses the requirements specified in the problem statement:
- ✅ English word playback timing accuracy
- ✅ Chinese character segmentation
- ✅ JavaScript playback precision
- ✅ Minimal code changes (surgical fixes only)
- ✅ Comprehensive testing
