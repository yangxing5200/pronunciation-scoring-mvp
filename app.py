import streamlit as st
import numpy as np
import time
import os
from pathlib import Path
import warnings

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
        
        return None
    
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
        
        return result


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
    
    # Sample challenges
    challenges = {
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
        st.info("🎵 Playing standard pronunciation...")
        st.caption("(In production, this plays pre-recorded native audio)")
    
    st.markdown("#### ✨ AI Voice Clone")
    st.caption("Hear this sentence in YOUR voice!")
    
    if st.button("🎨 Generate My Voice"):
        if "last_audio_path" in st.session_state:
            with st.spinner("Cloning your voice..."):
                cloned_path = st.session_state.processor.clone_voice(
                    st.session_state.last_audio_path,
                    target_text
                )
                
                if cloned_path:
                    st.success("✅ Generation Complete!")
                    st.audio(cloned_path)
                else:
                    st.warning("Voice cloning not available. Using fallback mode.")
                    st.caption("Install IndexTTS2 for voice cloning feature")
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
        delta = "good" if score >= 80 else "normal" if score >= 60 else "bad"
        st.metric("Overall Score", f"{score}/100", delta=delta)
    
    with m2:
        st.metric("🎯 Accuracy", f"{result['accuracy']}/100")
    
    with m3:
        st.metric("⚡ Fluency", f"{result['fluency']}/100")
    
    with m4:
        st.metric("🎵 Prosody", f"{result['prosody']}/100")
    
    # Word-level feedback
    st.markdown("### 📖 Word-by-Word Feedback")
    
    html_content = "<div style='font-size: 22px; line-height: 2.5; padding: 10px;'>"
    for w in result['word_scores']:
        score = w['score']
        word = w['word']
        
        # Color coding
        if score >= 90:
            color = "#d4edda"  # Green
            border = "#28a745"
        elif score >= 75:
            color = "#fff3cd"  # Yellow
            border = "#ffc107"
        else:
            color = "#f8d7da"  # Red
            border = "#dc3545"
        
        html_content += f"""
        <span style='
            background-color:{color}; 
            border: 2px solid {border};
            padding: 6px 12px; 
            border-radius: 8px; 
            margin: 4px;
            display: inline-block;
            font-weight: 500;
        '>
            {word} <small style='color: #666;'>{score}</small>
        </span>
        """
    html_content += "</div>"
    
    st.markdown(html_content, unsafe_allow_html=True)
    
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
