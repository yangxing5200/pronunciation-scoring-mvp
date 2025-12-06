import streamlit as st
import numpy as np
import time
import os
import re
from pathlib import Path
import warnings
import tempfile
import base64
import json
import html

# Import core modules
try:
    from core import (
        WhisperTranscriber,
        PhonemeAligner,
        PronunciationScorer,
        TextComparator,
        VoiceCloner
    )
    # Import Chinese-specific pipeline
    try:
        from core.chinese import ChineseScoringPipeline
        CHINESE_PIPELINE_AVAILABLE = True
    except ImportError:
        CHINESE_PIPELINE_AVAILABLE = False
        warnings.warn("Chinese scoring pipeline not available")
    
    CORE_AVAILABLE = True
except Exception as e:
    warnings.warn(f"Core modules not fully available: {e}")
    CORE_AVAILABLE = False
    CHINESE_PIPELINE_AVAILABLE = False

# Import audio recording component
try:
    from st_audiorec import st_audiorec
    AUDIOREC_AVAILABLE = True
except ImportError:
    warnings.warn("streamlit-audiorec not available. Using file upload only.")
    AUDIOREC_AVAILABLE = False


# Constants for audio playback timing
PLAYBACK_END_OFFSET = 0.01  # 10ms offset before end time


def find_word_timestamp(word, word_timestamps):
    """Find timestamp for a word by matching text instead of relying on index."""
    if not word_timestamps:
        return None
    
    word_lower = word.lower().strip()
    
    # Try exact match first
    for ts in word_timestamps:
        ts_word = ts.get('word', '').lower().strip()
        if ts_word == word_lower:
            return ts
    
    # Try fuzzy match
    for ts in word_timestamps:
        ts_word = ts.get('word', '').lower().strip()
        min_len = min(len(word_lower), len(ts_word))
        max_len = max(len(word_lower), len(ts_word))
        if max_len > 0 and min_len / max_len >= 0.7:
            if word_lower in ts_word or ts_word in word_lower:
                return ts
    
    return None


class AudioProcessor:
    """Main audio processing orchestrator."""
    
    def __init__(self):
        self.model_loaded = False
        self.transcriber = None
        self.scorer = None
        self.voice_cloner = None
        self.chinese_pipeline = None  # 中文专用管道
        
    def load_models(self):
        """Load all AI models."""
        if self.model_loaded:
            return
        
        if not CORE_AVAILABLE:
            st.error("Core modules not available. Please install dependencies.")
            return
        
        # 启动时清理超过 7 天的旧缓存
        self._cleanup_old_cache(max_age_days=7)
        
        try:
            # Check if WhisperX is available
            use_whisperx = False
            try:
                import whisperx
                use_whisperx = True
                print("WhisperX detected - will use enhanced alignment")
            except ImportError:
                print("WhisperX not available - using standard Whisper")
            
            # Initialize transcriber
            self.transcriber = WhisperTranscriber(
                model_size="base",
                model_dir="models/whisper",
                language="en",
                use_whisperx=use_whisperx
            )
            
            # Initialize scorer
            self.scorer = PronunciationScorer()
            
            # Initialize Chinese pipeline if available
            if CHINESE_PIPELINE_AVAILABLE:
                try:
                    self.chinese_pipeline = ChineseScoringPipeline(
                        device=getattr(self.transcriber, 'device', 'cpu')
                    )
                    self.chinese_pipeline.load_models(model_size="base")
                    print("✅ Chinese scoring pipeline loaded")
                except Exception as e:
                    warnings.warn(f"Chinese pipeline not available: {e}")
                    self.chinese_pipeline = None
            
            # Initialize voice cloner (optional)
            try:
                self.voice_cloner = VoiceCloner(
                    model_dir="models/indextts2"
                )
            except Exception as e:
                warnings.warn(f"Voice cloner not available: {e}")
                self.voice_cloner = None
            
            self.model_loaded = True
            
        except Exception as e:
            st.error(f"Failed to load models: {e}")
            st.info("Please run: python scripts/download_models.py")
            raise
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a short hash for text to use in filename."""
        import hashlib
        # 使用 MD5 的前 8 位作为哈希
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    
    def _get_standard_audio_path(self, text: str, language: str, voice_gender: str) -> Path:
        """Get the cached standard audio path for given text/language/gender."""
        output_dir = Path("temp_audio")
        output_dir.mkdir(exist_ok=True)
        
        # Determine language code
        is_chinese = language.startswith('zh') or bool(re.search(r'[\u4e00-\u9fff]', text))
        lang_code = "zh" if is_chinese else "en"
        
        # Generate filename with hash
        text_hash = self._get_text_hash(text)
        filename = f"standard_{voice_gender}_{lang_code}_{text_hash}.wav"
        
        return output_dir / filename
    
    def _cleanup_old_cache(self, max_age_days: int = 7):
        """Clean up old cached standard audio files."""
        import time
        output_dir = Path("temp_audio")
        if not output_dir.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for file in output_dir.glob("standard_*.wav"):
            try:
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age_seconds:
                    file.unlink()
                    print(f"🗑️ Cleaned old cache: {file.name}")
            except Exception:
                pass
    
    def generate_standard_audio(self, text, language="en", voice_gender="female", use_cache=True):
        """Generate standard pronunciation audio using IndexTTS2 with fixed reference speakers.
        
        Args:
            text: Text to synthesize
            language: Language code ('en' or 'zh')
            voice_gender: Voice gender ('female' or 'male')
            use_cache: If True, use cached audio if available
        
        Returns:
            Path to generated audio file, or None if failed
        """
        # Determine if Chinese
        is_chinese = language.startswith('zh') or bool(re.search(r'[\u4e00-\u9fff]', text))
        lang_code = "zh" if is_chinese else "en"
        
        # Get cached path
        output_path = self._get_standard_audio_path(text, language, voice_gender)
        
        # Check cache
        if use_cache and output_path.exists():
            print(f"✅ 使用缓存的标准音: {output_path.name} ({voice_gender} {lang_code})")
            return str(output_path)
        
        print(f"📢 生成新标准音: {output_path.name} ({voice_gender} {lang_code})")
        
        # Select reference audio based on language and gender
        ref_dir = Path("references")
        if is_chinese:
            ref_audio = ref_dir / f"standard_{voice_gender}_zh.wav"
        else:
            ref_audio = ref_dir / f"standard_{voice_gender}_en.wav"
        
        try:
            # METHOD 1: Use IndexTTS2 with fixed reference speaker (BEST QUALITY)
            if self.voice_cloner and self.voice_cloner.is_available():
                if ref_audio.exists():
                    print(f"🎙️ Generating standard audio using IndexTTS2 with {voice_gender} {language} reference")
                    success = self.voice_cloner.clone_voice(
                        text=text,
                        reference_audio_path=str(ref_audio),
                        output_path=output_path
                    )
                    
                    if success and output_path.exists():
                        print("✅ Standard audio generated successfully with IndexTTS2")
                        return str(output_path)
                else:
                    warnings.warn(f"Reference audio not found: {ref_audio}")
                    print(f"⚠️ Please add reference audio files to: {ref_dir}/")
                    print(f"   Required: standard_male_en.wav, standard_female_en.wav")
                    print(f"            standard_male_zh.wav, standard_female_zh.wav")
            
            # METHOD 2: Fallback to pyttsx3 (basic quality, offline)
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                
                if output_path.exists():
                    print("✅ Generated audio using pyttsx3 (fallback)")
                    return str(output_path)
            except Exception as e:
                print(f"pyttsx3 failed: {e}")
            
            return None
            
        except Exception as e:
            warnings.warn(f"TTS generation failed: {e}")
            return None
    
    def clone_voice(self, user_audio_path, standard_text):
        """Clone voice using IndexTTS2."""
        if self.voice_cloner and self.voice_cloner.is_available():
            output_path = Path("temp_audio") / "cloned_standard.wav"
            output_path.parent.mkdir(exist_ok=True)
            
            success = self.voice_cloner.clone_voice(
                text=standard_text,
                reference_audio_path=user_audio_path,
                output_path=output_path
            )
            
            if success:
                return str(output_path)
        
        # Fallback to standard TTS if voice cloning not available
        return self.generate_standard_audio(standard_text)
    
    def analyze_pronunciation(self, user_audio_file, reference_text, language="en", voice_gender="female"):
        """Core pronunciation analysis."""
        if not self.model_loaded:
            raise RuntimeError("Models not loaded")
        
        # Save uploaded audio to temp file
        temp_audio_path = Path("temp_audio") / "user_recording.wav"
        temp_audio_path.parent.mkdir(exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            f.write(user_audio_file.getvalue())
        
        # Detect if Chinese based on reference text
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', reference_text))
        
        # Set language for transcription
        transcription_language = "zh" if is_chinese else language
        
        # USE CHINESE PIPELINE FOR CHINESE TEXT
        if is_chinese and self.chinese_pipeline is not None:
            print("🇨🇳 Using specialized Chinese scoring pipeline...")
            try:
                # ========== 关键：根据语言和性别获取/生成标准音 ==========
                lang_code = "zh" if is_chinese else "en"
                
                # 获取缓存路径（包含语言、性别、文本哈希）
                standard_audio_path = self._get_standard_audio_path(
                    reference_text, 
                    lang_code, 
                    voice_gender
                )
                
                ref_audio_path = None
                
                # 检查缓存是否存在
                if standard_audio_path.exists():
                    ref_audio_path = str(standard_audio_path)
                    print(f"✅ 使用缓存标准音: {standard_audio_path.name}")
                else:
                    # 生成新的标准音
                    print(f"📢 生成标准音 (语言={lang_code}, 性别={voice_gender})...")
                    generated_path = self.generate_standard_audio(
                        reference_text,
                        language=lang_code,
                        voice_gender=voice_gender,
                        use_cache=False  # 已经检查过缓存了
                    )
                    if generated_path and Path(generated_path).exists():
                        ref_audio_path = generated_path
                        print(f"✅ 标准音生成成功: {Path(generated_path).name}")
                    else:
                        print(f"⚠️ 标准音生成失败，将使用模式分析评分（准确度降低）")
                
                # ========== 调用中文 Pipeline，传入标准音路径 ==========
                chinese_result = self.chinese_pipeline.score_pronunciation(
                    audio_path=str(temp_audio_path),
                    reference_text=reference_text,
                    reference_audio_path=ref_audio_path  # 传入正确的标准音！
                )
                
                # Convert Chinese pipeline result to standard format
                result = self._convert_chinese_result(chinese_result, temp_audio_path)
                return result
                
            except Exception as e:
                warnings.warn(f"Chinese pipeline failed, falling back: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to standard pipeline
        
        # STANDARD PIPELINE (for English or fallback)
        # Transcribe user audio
        transcription = self.transcriber.transcribe(
            str(temp_audio_path),
            language=transcription_language
        )
        
        # Store alignment type and phonemes for display
        alignment_type = transcription.get("alignment_type", "whisper")
        phonemes = transcription.get("phonemes", [])
        
        # Score pronunciation
        result = self.scorer.score_pronunciation(
            user_audio_path=str(temp_audio_path),
            reference_text=reference_text,
            transcribed_text=transcription["text"],
            word_timestamps=transcription["words"],
            reference_audio_path=None,
            language=transcription_language
        )
        
        # Store audio path, word timestamps, phonemes, and alignment info for word playback
        result['user_audio_path'] = str(temp_audio_path)
        result['word_timestamps'] = transcription["words"]
        result['phonemes'] = phonemes
        result['alignment_type'] = alignment_type
        
        return result
    
    def _convert_chinese_result(self, chinese_result: dict, audio_path: Path) -> dict:
        """Convert Chinese pipeline result to standard format for UI."""
        # Extract character scores and convert to word scores
        char_scores = chinese_result.get('character_scores', [])
        
        word_scores = []
        for char_data in char_scores:
            word_scores.append({
                'word': char_data.get('char', ''),
                'score': char_data.get('final_score', 70),
                'start': char_data.get('start', 0),
                'end': char_data.get('end', 0)
            })
        
        # Create word timestamps for playback
        word_timestamps = []
        for char_data in char_scores:
            word_timestamps.append({
                'word': char_data.get('char', ''),
                'start': char_data.get('start', 0),
                'end': char_data.get('end', 0),
                'probability': char_data.get('score', 1.0)
            })
        
        # Extract overall metrics
        overall_metrics = chinese_result.get('overall_metrics', {})
        
        # Map Chinese scores to standard format
        total_score = overall_metrics.get('overall_score', 70)
        accuracy = overall_metrics.get('avg_acoustic_score', 70)
        prosody = overall_metrics.get('avg_tone_score', 70)  # 声调 -> 韵律
        fluency = overall_metrics.get('avg_pause_score', 70)  # 流畅度
        
        # Generate issues from feedback
        issues = chinese_result.get('feedback', [])
        
        # Create text comparison
        reference_chars = [c.get('char', '') for c in char_scores]
        reference = ''.join(reference_chars)
        
        return {
            'total_score': total_score,
            'accuracy': accuracy,
            'fluency': fluency,
            'prosody': prosody,
            'word_scores': word_scores,
            'word_timestamps': word_timestamps,
            'issues': issues,
            'text_comparison': {
                'reference': reference,
                'hypothesis': reference,
                'similarity': 1.0 if total_score >= 80 else 0.8,
                'wer': 0.0,
                'missing_words': [],
                'extra_words': []
            },
            'user_audio_path': str(audio_path),
            'phonemes': [],
            'alignment_type': 'whisperx_chinese',
            'detailed_scores': {
                'acoustic': overall_metrics.get('avg_acoustic_score', 70),
                'tone': overall_metrics.get('avg_tone_score', 70),
                'duration': overall_metrics.get('avg_duration_score', 70),
                'pause': overall_metrics.get('avg_pause_score', 70)
            }
        }


@st.cache_resource
def load_audio_processor():
    """Load and cache the audio processor with all models."""
    print("=" * 60)
    print("Initializing AudioProcessor...")
    print("=" * 60)
    
    processor = AudioProcessor()
    processor.load_models()
    
    print("=" * 60)
    print("AudioProcessor initialization complete!")
    print("=" * 60)
    
    return processor


def load_practice_sentences():
    """Load practice sentences from JSON file."""
    sentences_file = Path(__file__).parent / "data" / "sentences.json"
    
    try:
        with open(sentences_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        warnings.warn(f"Sentences file not found: {sentences_file}")
        return {
            "English": {
                "Hello World": {
                    "text": "Hello world, this is a test.",
                    "phonetics": "/həˈloʊ wɜːrld ðɪs ɪz ə tɛst/",
                    "level": 1
                }
            },
            "Chinese": {
                "问候": {
                    "text": "你好",
                    "phonetics": "/ni3 hao3/",
                    "level": 1
                }
            }
        }
    except Exception as e:
        warnings.warn(f"Failed to load sentences: {e}")
        return {"English": {}, "Chinese": {}}


# === Streamlit UI ===

st.set_page_config(page_title="AI Pronunciation Coach MVP", layout="wide")

st.title("🎙️ AI Pronunciation Coach")
st.markdown("### Personal AI Spoken Language Tutor - **Fully Offline**")

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    language = st.selectbox("Target Language", ["English", "Chinese"])
    difficulty = st.slider("Difficulty Level", 1, 5, 2)
    
    st.markdown("### 🎙️ Standard Voice")
    voice_gender = st.radio(
        "Reference Voice", 
        options=["female", "male"], 
        index=0,
        horizontal=True,
        key="voice_gender_selector"
    )
    st.session_state.voice_gender = voice_gender
    st.caption(f"✓ Using {voice_gender} voice")
    
    st.divider()
    st.markdown("### 🤖 System Status")
    
    # Load processor using cached function
    try:
        processor = load_audio_processor()
        st.session_state.processor = processor
        st.success("✅ AI Engine Ready")
        
        # Show loaded components
        st.markdown("**Loaded Components:**")
        st.markdown("- ✅ Whisper Transcriber")
        
        # Check if WhisperX is available
        if hasattr(processor.transcriber, 'use_whisperx') and \
           processor.transcriber.use_whisperx:
            st.markdown("- ✨ WhisperX Enhanced Alignment")
            st.markdown("  - Word-level for Chinese")
            st.markdown("  - Phoneme-level for English")
        else:
            st.markdown("- ⚠️ WhisperX (not installed)")
            st.caption("Install for better accuracy")
        
        st.markdown("- ✅ Pronunciation Scorer")
        
        # Show Chinese pipeline status
        if CHINESE_PIPELINE_AVAILABLE and processor.chinese_pipeline:
            st.markdown("- ✅ Chinese Scoring Pipeline")
            st.markdown("  - 声调评分 (Tone scoring)")
            st.markdown("  - 韵母评分 (Final scoring)")
            st.markdown("  - 流畅度评分 (Fluency)")
        elif language == "Chinese":
            st.markdown("- ⚠️ Chinese Pipeline (basic mode)")
            st.caption("Install transformers for advanced scoring")
        
        if processor.voice_cloner and processor.voice_cloner.is_available():
            st.markdown("- ✅ Voice Cloner")
        else:
            st.markdown("- ⚠️ Voice Cloner (fallback mode)")
            
    except Exception as e:
        st.error(f"❌ Failed to load models: {e}")
        st.info("Run: `python scripts/download_models.py`")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    st.divider()
    st.markdown("### 📊 About")
    st.markdown("""
    **Offline-First Design**
    - All processing runs locally
    - No internet required
    - Privacy-focused
    
    **Scoring Dimensions:**
    - 🎯 Accuracy
    - ⚡ Fluency  
    - 🎵 Prosody
    """)

# Main Area
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Challenge Card")
    
    all_sentences = load_practice_sentences()
    challenges = all_sentences.get(language, {})
    
    if not challenges:
        st.warning(f"No practice sentences available for {language}")
        challenges = {
            "Default": {
                "text": "No sentences available.",
                "phonetics": "",
                "level": 1
            }
        }
    
    selected_challenge = st.selectbox("Choose Challenge", list(challenges.keys()))
    
    challenge = challenges[selected_challenge]
    target_text = challenge["text"]
    phonetics = challenge["phonetics"]
    
    st.info(f"**Target:** {target_text}")
    st.code(phonetics, language="text")
    
    st.markdown("#### 🔊 Standard Audio")
    st.caption(f"🔍 Current settings - Language: {language}, Gender: {voice_gender}")
    
    if st.button("▶️ Play Standard (Native)"):
        if "processor" in st.session_state:
            with st.spinner("Loading standard pronunciation..."):
                lang_code = "zh" if language == "Chinese" else "en"
                
                # 检查是否有缓存
                cached_path = st.session_state.processor._get_standard_audio_path(
                    target_text, lang_code, voice_gender
                )
                is_cached = cached_path.exists()
                
                audio_path = st.session_state.processor.generate_standard_audio(
                    target_text, 
                    language=lang_code,
                    voice_gender=voice_gender
                )
                
                if audio_path and Path(audio_path).exists():
                    if is_cached:
                        st.success(f"✅ 使用缓存 ({voice_gender} {lang_code})")
                    else:
                        st.success(f"✅ 已生成标准音 ({voice_gender} {lang_code})")
                    st.audio(audio_path)
                else:
                    expected_ref = f"references/standard_{voice_gender}_{lang_code}.wav"
                    st.warning("⚠️ IndexTTS2 not available or reference missing")
                    st.error(f"❌ Could not find: {expected_ref}")
        else:
            st.error("Models not loaded yet!")
    
    st.markdown("#### ✨ AI Voice Clone")
    st.caption("Hear this sentence in YOUR voice!")
    
    if st.button("🎨 Generate My Voice"):
        if "processor" not in st.session_state:
            st.error("Models not loaded yet!")
        elif "last_audio_path" in st.session_state:
            with st.spinner("Cloning your voice..."):
                cloned_path = st.session_state.processor.clone_voice(
                    st.session_state.last_audio_path,
                    target_text
                )
                
                if cloned_path and Path(cloned_path).exists():
                    st.success("✅ Generation Complete!")
                    st.audio(cloned_path)
                else:
                    st.warning("⚠️ Voice cloning not available.")
        else:
            st.error("⚠️ Please record or upload audio first!")

with col2:
    st.subheader("🎤 Practice Area")
    
    st.markdown("#### Record Your Pronunciation")
    
    audio_file = None
    
    if AUDIOREC_AVAILABLE:
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            st.success("✅ Recording captured!")
            
            temp_path = Path("temp_audio") / "recorded.wav"
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(wav_audio_data)
            
            st.session_state.last_audio_path = str(temp_path)
            
            import io
            audio_file = io.BytesIO(wav_audio_data)
            audio_file.name = "recording.wav"
    else:
        st.info("💡 Tip: Install streamlit-audiorec for one-click recording")
    
    st.markdown("#### Or Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload Recording (.wav, .mp3)",
        type=['wav', 'mp3'],
        help="Upload your pronunciation recording"
    )
    
    if uploaded_file is not None:
        audio_file = uploaded_file
        st.success("✅ Audio uploaded successfully!")
        
        temp_path = Path("temp_audio") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.last_audio_path = str(temp_path)
        
        st.audio(uploaded_file)
    
    if audio_file is not None:
        if st.button("🔍 Analyze Pronunciation", type="primary"):
            if "processor" not in st.session_state or not st.session_state.processor.model_loaded:
                st.error("❌ Models not loaded. Please check sidebar.")
            else:
                with st.spinner("🔬 Analyzing pronunciation..."):
                    try:
                        # 获取 voice_gender 参数
                        voice_gender = st.session_state.get('voice_gender', 'female')
                        
                        result = st.session_state.processor.analyze_pronunciation(
                            audio_file,
                            target_text,
                            language=language.lower()[:2],
                            voice_gender=voice_gender  # 传入性别参数！
                        )
                        
                        st.session_state.last_result = result
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

# Display results
if "last_result" in st.session_state:
    result = st.session_state.last_result
    
    st.divider()
    st.markdown("## 📊 Analysis Report")
    
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        score = result['total_score']
        st.metric("Overall Score", f"{score}/100")
    
    with m2:
        st.metric("🎯 Accuracy", f"{result['accuracy']}/100")
    
    with m3:
        st.metric("⚡ Fluency", f"{result['fluency']}/100")
    
    with m4:
        st.metric("🎵 Prosody", f"{result['prosody']}/100")
    
    # Word-level feedback
    st.markdown("### 📖 Word-by-Word Feedback")
    
    alignment_type = result.get('alignment_type', 'whisper')
    if alignment_type == 'whisperx_chinese':
        st.caption("✨ Using Chinese specialized scoring pipeline")
    elif alignment_type == 'whisperx':
        st.caption("✨ Using WhisperX enhanced alignment")
    else:
        st.caption("💡 Tip: Install WhisperX for improved accuracy")
    
    st.caption("Click on any word to hear your pronunciation")
    
    word_timestamps = result.get('word_timestamps', [])
    user_audio_path = result.get('user_audio_path', None)
    word_scores = result['word_scores']
    
    if user_audio_path and Path(user_audio_path).exists():
        with open(user_audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        word_html = ""
        for idx, w in enumerate(word_scores):
            word = w['word']
            score = w['score']
            
            word_ts = find_word_timestamp(word, word_timestamps)
            
            if word_ts:
                start_time = word_ts.get('start', 0)
                end_time = word_ts.get('end', 0)
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    if start_time >= end_time or start_time < 0:
                        start_time = -1
                        end_time = -1
                except (TypeError, ValueError):
                    start_time = -1
                    end_time = -1
            else:
                start_time = -1
                end_time = -1
            
            try:
                score_val = float(score)
                if score_val >= 90:
                    color = "#28a745"
                    emoji = "✅"
                elif score_val >= 75:
                    color = "#ffc107"
                    emoji = "⚠️"
                else:
                    color = "#dc3545"
                    emoji = "❌"
            except (TypeError, ValueError):
                color = "#6c757d"
                emoji = "❓"
            
            word_escaped = html.escape(str(word))
            emoji_escaped = html.escape(str(emoji))
            score_escaped = html.escape(str(score))
            
            if start_time >= 0 and end_time > start_time:
                word_html += f'''
                <button onclick="playWord({start_time}, {end_time})" 
                        style="margin:4px; padding:8px 12px; border-radius:8px; 
                               border:2px solid {color}; background:white; 
                               cursor:pointer; font-size:14px;">
                    {emoji_escaped} {word_escaped}<br><small>{score_escaped}</small>
                </button>
                '''
            else:
                word_html += f'''
                <button disabled 
                        style="margin:4px; padding:8px 12px; border-radius:8px; 
                               border:2px solid {color}; background:#f0f0f0; 
                               cursor:not-allowed; font-size:14px; opacity:0.6;">
                    {emoji_escaped} {word_escaped}<br><small>{score_escaped}</small>
                </button>
                '''
        
        st.components.v1.html(f'''
        <audio id="user-recording" src="data:audio/wav;base64,{audio_base64}" style="display:none;"></audio>
        <script>
            let animationFrameId = null;
            const PLAYBACK_END_OFFSET = {PLAYBACK_END_OFFSET};
            
            function playWord(startTime, endTime) {{
                const audio = document.getElementById('user-recording');
                audio.pause();
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                
                function checkTime() {{
                    if (audio.currentTime >= endTime - PLAYBACK_END_OFFSET) {{
                        audio.pause();
                        animationFrameId = null;
                    }} else if (!audio.paused) {{
                        animationFrameId = requestAnimationFrame(checkTime);
                    }}
                }}
                
                function attemptPlayback() {{
                    if (audio.readyState >= 2) {{
                        audio.currentTime = startTime;
                        audio.play().then(function() {{
                            animationFrameId = requestAnimationFrame(checkTime);
                        }}).catch(function(error) {{
                            console.error('Playback failed:', error);
                        }});
                    }} else {{
                        audio.addEventListener('loadeddata', function onLoaded() {{
                            audio.removeEventListener('loadeddata', onLoaded);
                            audio.currentTime = startTime;
                            audio.play().then(function() {{
                                animationFrameId = requestAnimationFrame(checkTime);
                            }}).catch(function(error) {{
                                console.error('Playback failed:', error);
                            }});
                        }});
                    }}
                }}
                
                attemptPlayback();
            }}
        </script>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            {word_html}
        </div>
        ''', height=200)
    else:
        st.warning("Audio file not available for word playback.")
    
    # Issues and coaching tips
    st.markdown("### 💡 Coaching Tips & Issues")
    
    if result['issues']:
        for i, issue in enumerate(result['issues'], 1):
            if i == 1:
                st.error(f"🔴 **Priority {i}:** {issue}")
            else:
                st.warning(f"⚠️ **Issue {i}:** {issue}")
    else:
        st.success("🎉 Excellent! No major issues detected.")
    
    # Detailed breakdown
    with st.expander("📈 Detailed Breakdown"):
        st.markdown("#### Text Comparison")
        tc = result['text_comparison']
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Expected:**")
            st.code(tc.get('reference', ''))
        with col_b:
            st.markdown("**You said:**")
            st.code(tc.get('hypothesis', ''))
        
        st.markdown(f"**Similarity:** {tc.get('similarity', 0) * 100:.1f}%")
        st.markdown(f"**Word Error Rate:** {tc.get('wer', 0) * 100:.1f}%")
        
        if tc.get('missing_words'):
            st.warning(f"Missing words: {', '.join(tc['missing_words'])}")
        if tc.get('extra_words'):
            st.info(f"Extra words: {', '.join(tc['extra_words'])}")
        
        # Display Chinese-specific scores if available
        if 'detailed_scores' in result:
            st.markdown("#### Chinese Pronunciation Details")
            detailed = result['detailed_scores']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎤 声母韵母", f"{detailed.get('acoustic', 0)}/100")
            with col2:
                st.metric("🎵 声调", f"{detailed.get('tone', 0)}/100")
            with col3:
                st.metric("⏱️ 时长", f"{detailed.get('duration', 0)}/100")
            with col4:
                st.metric("🌊 流畅度", f"{detailed.get('pause', 0)}/100")
        
        # Display phoneme information if available (WhisperX for English)
        phonemes = result.get('phonemes', [])
        if phonemes:
            st.markdown("#### Phoneme-Level Analysis")
            st.caption("✨ Enhanced phoneme-level timestamps from WhisperX")
            st.markdown(f"**Total phonemes detected:** {len(phonemes)}")
            
            if len(phonemes) > 0:
                sample_size = min(10, len(phonemes))
                st.markdown(f"**Sample phonemes (first {sample_size}):**")
                phoneme_data = []
                for p in phonemes[:sample_size]:
                    phoneme_data.append({
                        "Phoneme": p.get('phoneme', ''),
                        "Word": p.get('word', ''),
                        "Start": f"{p.get('start', 0):.3f}s",
                        "End": f"{p.get('end', 0):.3f}s"
                    })
                st.table(phoneme_data)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 <strong>Fully Offline AI Pronunciation Coach</strong></p>
    <p>All processing runs locally on your machine • Privacy-first design</p>
    <p><small>Powered by Whisper, IndexTTS2, WavLM, and Chinese Pipeline</small></p>
</div>
""", unsafe_allow_html=True)