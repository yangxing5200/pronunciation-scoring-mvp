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
    CORE_AVAILABLE = True
except Exception as e:
    warnings.warn(f"Core modules not fully available: {e}")
    CORE_AVAILABLE = False

# Import audio recording component
try:
    from st_audiorec import st_audiorec
    AUDIOREC_AVAILABLE = True
except ImportError:
    warnings.warn("streamlit-audiorec not available. Using file upload only.")
    AUDIOREC_AVAILABLE = False

# Import TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    warnings.warn("pyttsx3 not available. TTS will use fallback mode.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    warnings.warn("gTTS not available. Online TTS fallback disabled.")


# Constants for audio playback timing
# This offset prevents playing into the next word due to requestAnimationFrame timing (~60fps = ~16ms)
PLAYBACK_END_OFFSET = 0.01  # 10ms offset before end time


def find_word_timestamp(word, word_timestamps):
    """
    Find timestamp for a word by matching text instead of relying on index.
    
    Args:
        word: Word text to find
        word_timestamps: List of timestamp dictionaries from Whisper
    
    Returns:
        Dictionary with word, start, end, probability or None if not found
    """
    if not word_timestamps:
        return None
    
    # Clean the word for matching
    word_lower = word.lower().strip()
    
    # Try exact match first
    for ts in word_timestamps:
        ts_word = ts.get('word', '').lower().strip()
        if ts_word == word_lower:
            return ts
    
    # Try fuzzy match (word contained in timestamp or vice versa)
    # Only match if words are similar length to avoid false positives like 'he' matching 'hello'
    for ts in word_timestamps:
        ts_word = ts.get('word', '').lower().strip()
        # Require that the shorter word is at least 50% of the longer word's length
        # This prevents 'he' from matching 'hello' or 'the'
        min_len = min(len(word_lower), len(ts_word))
        max_len = max(len(word_lower), len(ts_word))
        if max_len > 0 and min_len / max_len >= 0.5:
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
        
    def load_models(self):
        """Load all AI models."""
        if self.model_loaded:
            return
        
        if not CORE_AVAILABLE:
            st.error("Core modules not available. Please install dependencies.")
            return
        
        try:
            # Initialize transcriber
            self.transcriber = WhisperTranscriber(
                model_size="base",
                model_dir="models/whisper",
                language="en"
            )
            
            # Initialize scorer
            self.scorer = PronunciationScorer()
            
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
    
    def generate_standard_audio(self, text):
        """Generate standard pronunciation audio using TTS."""
        output_dir = Path("temp_audio")
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Try pyttsx3 first (offline)
            if PYTTSX3_AVAILABLE:
                output_path = output_dir / "standard_pronunciation.wav"
                engine = pyttsx3.init()
                # Set properties for better quality
                engine.setProperty('rate', 150)  # Speed of speech
                engine.setProperty('volume', 0.9)
                
                # Save to file
                engine.save_to_file(text, str(output_path))
                engine.runAndWait()
                
                if output_path.exists():
                    return str(output_path)
            
            # Fallback to gTTS (requires internet)
            if GTTS_AVAILABLE:
                output_path = output_dir / "standard_pronunciation.mp3"
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(str(output_path))
                
                if output_path.exists():
                    return str(output_path)
            
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
    
    def analyze_pronunciation(self, user_audio_file, reference_text):
        """Core pronunciation analysis."""
        if not self.model_loaded:
            raise RuntimeError("Models not loaded")
        
        # Save uploaded audio to temp file
        temp_audio_path = Path("temp_audio") / "user_recording.wav"
        temp_audio_path.parent.mkdir(exist_ok=True)
        
        with open(temp_audio_path, "wb") as f:
            f.write(user_audio_file.getvalue())
        
        # Transcribe user audio
        transcription = self.transcriber.transcribe(str(temp_audio_path))
        
        # Score pronunciation
        result = self.scorer.score_pronunciation(
            user_audio_path=str(temp_audio_path),
            reference_text=reference_text,
            transcribed_text=transcription["text"],
            word_timestamps=transcription["words"],
            reference_audio_path=None  # Could add reference audio
        )
        
        # Store audio path and word timestamps for word playback
        result['user_audio_path'] = str(temp_audio_path)
        result['word_timestamps'] = transcription["words"]
        
        return result


def load_practice_sentences():
    """Load practice sentences from JSON file."""
    sentences_file = Path(__file__).parent / "data" / "sentences.json"
    
    try:
        with open(sentences_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        warnings.warn(f"Sentences file not found: {sentences_file}")
        # Return default English sentences as fallback
        return {
            "English": {
                "Hello World": {
                    "text": "Hello world, this is a test.",
                    "phonetics": "/həˈloʊ wɜːrld ðɪs ɪz ə tɛst/",
                    "level": 1
                },
                "Weather Talk": {
                    "text": "The weather is beautiful today.",
                    "phonetics": "/ðə ˈwɛðər ɪz ˈbjutəfəl təˈdeɪ/",
                    "level": 2
                },
                "Technology": {
                    "text": "Artificial intelligence is transforming the world.",
                    "phonetics": "/ˌɑrtəˈfɪʃəl ɪnˈtɛlɪdʒəns ɪz trænsˈfɔrmɪŋ ðə wɜrld/",
                    "level": 3
                }
            },
            "Chinese": {}
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
    
    st.divider()
    st.markdown("### 🤖 System Status")
    
    if "processor" not in st.session_state:
        with st.spinner("Initializing AI Engine..."):
            st.session_state.processor = AudioProcessor()
            try:
                st.session_state.processor.load_models()
                st.success("✅ AI Engine Ready")
            except Exception as e:
                st.error(f"❌ Failed to load: {e}")
                st.info("Run: `python scripts/download_models.py`")
    else:
        if st.session_state.processor.model_loaded:
            st.success("✅ AI Engine Ready")
            
            # Show loaded components
            st.markdown("**Loaded Components:**")
            st.markdown("- ✅ Whisper Transcriber")
            st.markdown("- ✅ Pronunciation Scorer")
            
            if st.session_state.processor.voice_cloner and \
               st.session_state.processor.voice_cloner.is_available():
                st.markdown("- ✅ Voice Cloner")
            else:
                st.markdown("- ⚠️ Voice Cloner (fallback mode)")
        else:
            st.warning("⚠️ Models not loaded")
    
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
    
    # Load practice sentences from JSON
    all_sentences = load_practice_sentences()
    
    # Filter by selected language
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
    
    selected_challenge = st.selectbox(
        "Choose Challenge",
        list(challenges.keys())
    )
    
    challenge = challenges[selected_challenge]
    target_text = challenge["text"]
    phonetics = challenge["phonetics"]
    
    st.info(f"**Target:** {target_text}")
    st.code(phonetics, language="text")
    
    st.markdown("#### 🔊 Standard Audio")
    if st.button("▶️ Play Standard (Native)"):
        with st.spinner("Generating standard pronunciation..."):
            audio_path = st.session_state.processor.generate_standard_audio(target_text)
            
            if audio_path and Path(audio_path).exists():
                st.success("✅ Audio generated!")
                st.audio(audio_path)
            else:
                st.warning("⚠️ TTS not available. Please install pyttsx3 or gTTS:")
                st.code("pip install pyttsx3 gTTS", language="bash")
                st.info("📝 Standard pronunciation text:")
                st.markdown(f"**{target_text}**")
    
    st.markdown("#### ✨ AI Voice Clone")
    st.caption("Hear this sentence in YOUR voice!")
    
    if st.button("🎨 Generate My Voice"):
        if "last_audio_path" in st.session_state:
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
                    st.info("📌 Using standard TTS as fallback. Install IndexTTS2 for true voice cloning.")
        else:
            st.error("⚠️ Please record or upload audio first!")

with col2:
    st.subheader("🎤 Practice Area")
    
    # Recording section
    st.markdown("#### Record Your Pronunciation")
    
    audio_file = None
    
    # Try to use streamlit-audiorec if available
    if AUDIOREC_AVAILABLE:
        wav_audio_data = st_audiorec()
        
        if wav_audio_data is not None:
            st.success("✅ Recording captured!")
            
            # Save to session state
            temp_path = Path("temp_audio") / "recorded.wav"
            temp_path.parent.mkdir(exist_ok=True)
            temp_path.write_bytes(wav_audio_data)
            
            st.session_state.last_audio_path = str(temp_path)
            
            # Create a file-like object for processing
            import io
            audio_file = io.BytesIO(wav_audio_data)
            audio_file.name = "recording.wav"
    else:
        st.info("💡 Tip: Install streamlit-audiorec for one-click recording")
    
    # File upload as alternative/backup
    st.markdown("#### Or Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload Recording (.wav, .mp3)",
        type=['wav', 'mp3'],
        help="Upload your pronunciation recording"
    )
    
    if uploaded_file is not None:
        audio_file = uploaded_file
        st.success("✅ Audio uploaded successfully!")
        
        # Save for voice cloning
        temp_path = Path("temp_audio") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state.last_audio_path = str(temp_path)
        
        # Show audio player
        st.audio(uploaded_file)
    
    # Analysis button
    if audio_file is not None:
        if st.button("🔍 Analyze Pronunciation", type="primary"):
            if not st.session_state.processor.model_loaded:
                st.error("❌ Models not loaded. Please check sidebar for status.")
            else:
                with st.spinner("🔬 Analyzing phonemes, pitch, and rhythm..."):
                    try:
                        result = st.session_state.processor.analyze_pronunciation(
                            audio_file,
                            target_text
                        )
                        
                        # Store result in session state
                        st.session_state.last_result = result
                        
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {e}")
                        st.info("Please check that audio file is valid and models are loaded.")

# Display results if available
if "last_result" in st.session_state:
    result = st.session_state.last_result
    
    st.divider()
    st.markdown("## 📊 Analysis Report")
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        score = result['total_score']
        # Determine color based on score
        if score >= 80:
            delta_color = "normal"
        elif score >= 60:
            delta_color = "off"
        else:
            delta_color = "inverse"
        st.metric("Overall Score", f"{score}/100")
    
    with m2:
        st.metric("🎯 Accuracy", f"{result['accuracy']}/100")
    
    with m3:
        st.metric("⚡ Fluency", f"{result['fluency']}/100")
    
    with m4:
        st.metric("🎵 Prosody", f"{result['prosody']}/100")
    
    # Word-level feedback
    st.markdown("### 📖 Word-by-Word Feedback")
    st.caption("Click on any word to hear your pronunciation of that word")
    
    # Get word timestamps if available
    word_timestamps = result.get('word_timestamps', [])
    user_audio_path = result.get('user_audio_path', None)
    
    # Display words using JavaScript and HTML5 Audio API
    word_scores = result['word_scores']
    
    if user_audio_path and Path(user_audio_path).exists():
        # Read audio file and convert to base64
        with open(user_audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode()
        
        # Build HTML for word buttons
        word_html = ""
        for idx, w in enumerate(word_scores):
            word = w['word']
            score = w['score']
            
            # Find timestamp by matching word text instead of using index
            word_ts = find_word_timestamp(word, word_timestamps)
            
            if word_ts:
                start_time = word_ts.get('start', 0)
                end_time = word_ts.get('end', 0)
                # Validate timestamps are numeric
                try:
                    start_time = float(start_time)
                    end_time = float(end_time)
                    # Skip words with invalid timestamps
                    if start_time >= end_time or start_time < 0:
                        start_time = -1
                        end_time = -1
                except (TypeError, ValueError):
                    start_time = -1
                    end_time = -1
            else:
                # Mark words without timestamps as disabled
                start_time = -1
                end_time = -1
            
            # Color coding based on score (using safe predefined colors)
            # Validate score is numeric for safe color selection
            try:
                score_val = float(score)
                if score_val >= 90:
                    color = "#28a745"  # green
                    emoji = "✅"
                elif score_val >= 75:
                    color = "#ffc107"  # yellow
                    emoji = "⚠️"
                else:
                    color = "#dc3545"  # red
                    emoji = "❌"
            except (TypeError, ValueError):
                # Fallback color for invalid scores
                color = "#6c757d"  # gray
                emoji = "❓"
            
            # Escape word and emoji to prevent XSS
            word_escaped = html.escape(str(word))
            emoji_escaped = html.escape(str(emoji))
            score_escaped = html.escape(str(score))
            
            # Only create playable button if timestamps are valid
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
                # Disabled button for words without timestamps
                word_html += f'''
                <button disabled 
                        style="margin:4px; padding:8px 12px; border-radius:8px; 
                               border:2px solid {color}; background:#f0f0f0; 
                               cursor:not-allowed; font-size:14px; opacity:0.6;">
                    {emoji_escaped} {word_escaped}<br><small>{score_escaped}</small>
                </button>
                '''
        
        # Render HTML component with audio player and JavaScript
        st.components.v1.html(f'''
        <audio id="user-recording" src="data:audio/wav;base64,{audio_base64}" style="display:none;"></audio>
        <script>
            let animationFrameId = null;
            const PLAYBACK_END_OFFSET = {PLAYBACK_END_OFFSET};  // Offset to prevent playing next word
            
            function playWord(startTime, endTime) {{
                const audio = document.getElementById('user-recording');
                
                // Stop any currently playing audio and cancel animation frame
                audio.pause();
                if (animationFrameId) {{
                    cancelAnimationFrame(animationFrameId);
                    animationFrameId = null;
                }}
                
                // Function to check playback position
                function checkTime() {{
                    if (audio.currentTime >= endTime - PLAYBACK_END_OFFSET) {{
                        // Stop slightly before end to avoid playing next word
                        audio.pause();
                        animationFrameId = null;
                    }} else if (!audio.paused) {{
                        // Continue checking
                        animationFrameId = requestAnimationFrame(checkTime);
                    }}
                }}
                
                // Wait for audio to be ready before setting currentTime
                function attemptPlayback() {{
                    if (audio.readyState >= 2) {{
                        // Audio has loaded enough data
                        audio.currentTime = startTime;
                        
                        // Play with error handling
                        audio.play().then(function() {{
                            // Start checking playback position
                            animationFrameId = requestAnimationFrame(checkTime);
                        }}).catch(function(error) {{
                            console.error('Playback failed:', error);
                            // Audio playback might fail if user hasn't interacted with the page yet
                        }});
                    }} else {{
                        // Wait for audio to load
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

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🤖 <strong>Fully Offline AI Pronunciation Coach</strong></p>
    <p>All processing runs locally on your machine • Privacy-first design</p>
    <p><small>Powered by Whisper, IndexTTS2, librosa, and DTW</small></p>
</div>
""", unsafe_allow_html=True)
