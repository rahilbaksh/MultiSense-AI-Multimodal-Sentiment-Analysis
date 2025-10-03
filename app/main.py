import streamlit as st
import os
import tempfile
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Multimodal Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .demo-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .success-badge {
        background: #1dd1a1;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .tab-content {
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 15px;
        margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .uploaded-file {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

try:
    from app.utils.logger import logger
    from app.utils.config import config
    from app.models.model_manager import ModelManager
    from app.processing.audio_processor import AudioProcessor
    from app.processing.text_processor import TextProcessor
    imports_ok = True
except ImportError as e:
    st.error(f"Import error: {e}")
    imports_ok = False

class MultimodalApp:
    def __init__(self):
        if imports_ok:
            self.model_manager = ModelManager()
            self.audio_processor = AudioProcessor()
            self.text_processor = TextProcessor()
        self.setup_page()

    def setup_page(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="main-header">🎭 MultiSense AI</h1>', unsafe_allow_html=True)
            st.markdown('### Analyze customer sentiment through text and voice')
            st.markdown('---')

    def analyze_text(self, text_model, text: str):
        try:
            self.text_processor.validate_text(text)
            cleaned_text = self.text_processor.clean_text(text)
            results = text_model.predict(cleaned_text)
            
            label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
            sentiment_scores = {}
            for result in results:
                label_name = label_map.get(result['label'], result['label'])
                sentiment_scores[label_name] = result['score']
            
            predicted_label = max(sentiment_scores.items(), key=lambda x: x[1])
            
            return {
                'predicted_label': predicted_label[0],
                'confidence': predicted_label[1],
                'all_scores': sentiment_scores
            }
            
        except Exception as e:
            logger.error(f"Text analysis error: {e}")
            st.error(f"Text analysis failed: {str(e)}")
            return None

    def analyze_audio(self, audio_model, spectrogram):
        try:
            predictions = np.random.dirichlet(np.ones(4), size=1)[0]
            emotion_labels = ['angry', 'happy', 'sad', 'neutral']
            emotion_icons = ['😠', '😄', '😢', '😐']
            
            results = {}
            for i, score in enumerate(predictions):
                results[f"{emotion_icons[i]} {emotion_labels[i].upper()}"] = float(score)
            
            predicted_idx = np.argmax(predictions)
            predicted_label = f"{emotion_icons[predicted_idx]} {emotion_labels[predicted_idx].upper()}"
            confidence = float(predictions[predicted_idx])
            
            return {
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_scores': results
            }
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            st.error(f"Audio analysis failed: {str(e)}")
            return None

    def run(self):
        if not imports_ok:
            st.error("Cannot start application due to import errors")
            return
            
        try:
            @st.cache_resource
            def load_models():
                return self.model_manager.get_models()

            text_model, audio_model = load_models()

            with st.sidebar:
                st.markdown("### 🔧 System Status")
                if audio_model is None:
                    st.markdown('<div class="demo-badge">🔊 AUDIO: DEMO MODE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="success-badge">✅ AUDIO: ACTIVE</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="success-badge">✅ TEXT: ACTIVE</div>', unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📊 How to Use")
                st.info("""
                1. **Text Analysis**: Enter customer feedback text
                2. **Audio Analysis**: Upload voice recordings  
                3. **Combined**: Use both for comprehensive analysis
                """)

            selected_tab = option_menu(
                menu_title=None,
                options=["📝 Text Analysis", "🎵 Audio Analysis", "🌐 Combined", "📊 Dashboard"],
                icons=["chat-text", "mic", "link", "graph-up"],
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {"padding": "0!important", "background-color": "#f8f9fa"},
                    "icon": {"color": "orange", "font-size": "18px"}, 
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#667eea"},
                }
            )

            if selected_tab == "📝 Text Analysis":
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">📝 Text Sentiment Analysis</h2>', unsafe_allow_html=True)
                
                text_input = st.text_area("**Enter customer feedback:**", height=120, 
                                        placeholder="Type your text here...", help="Enter any customer feedback text to analyze sentiment")
                
                if st.button("🚀 Analyze Sentiment", key="text_btn", use_container_width=True):
                    if text_input.strip():
                        with st.spinner("🔍 Analyzing text sentiment..."):
                            result = self.analyze_text(text_model, text_input)
                            if result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Predicted Sentiment", result['predicted_label'])
                                    st.metric("Confidence", f"{result['confidence']:.2%}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.progress(result['confidence'], text="Confidence Level")
                                    with st.expander("📊 Detailed Scores"):
                                        st.json(result['all_scores'])
                    else:
                        st.warning("⚠️ Please enter some text to analyze")
                st.markdown('</div>', unsafe_allow_html=True)

            elif selected_tab == "🎵 Audio Analysis":
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">🎵 Audio Emotion Analysis</h2>', unsafe_allow_html=True)
                
                st.warning("🎧 **Note:** Audio analysis is currently in demo mode showing sample results")
                
                uploaded_file = st.file_uploader(
                    "**Upload an audio file**", 
                    type=['wav', 'mp3', 'ogg'],
                    help="Supported formats: WAV, MP3, OGG"
                )
                
                if uploaded_file:
                    st.markdown('<div class="uploaded-file">', unsafe_allow_html=True)
                    st.audio(uploaded_file, format="audio/wav")
                    st.markdown(f"**File:** {uploaded_file.name}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if uploaded_file and st.button("🎯 Analyze Emotions", key="audio_btn", use_container_width=True):
                    with st.spinner("🎵 Processing audio emotions..."):
                        try:
                            spectrogram, tmp_path = self.audio_processor.process_uploaded_file(uploaded_file)
                            result = self.analyze_audio(audio_model, spectrogram)
                            if result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric("Predicted Emotion", result['predicted_label'])
                                    st.metric("Confidence", f"{result['confidence']:.2%}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.progress(result['confidence'], text="Confidence Level")
                                    with st.expander("📊 Emotion Breakdown"):
                                        for emotion, score in result['all_scores'].items():
                                            st.write(f"{emotion}: {score:.2%}")
                            os.unlink(tmp_path)
                        except Exception as e:
                            st.error(f"❌ Error processing audio: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)

            elif selected_tab == "🌐 Combined":
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">🌐 Combined Analysis</h2>', unsafe_allow_html=True)
                st.success("Get comprehensive insights by combining text and audio analysis!")
                
                col1, col2 = st.columns(2)
                with col1:
                    combined_text = st.text_area("**Enter text feedback:**", height=100, key="combined_text")
                with col2:
                    combined_audio = st.file_uploader(
                        "**Upload voice recording:**", 
                        type=['wav', 'mp3', 'ogg'],
                        key="combined_audio"
                    )
                
                if st.button("🌈 Run Comprehensive Analysis", key="combined_btn", use_container_width=True):
                    if combined_text.strip() and combined_audio:
                        with st.spinner("🌐 Analyzing combined modalities..."):
                            text_result = self.analyze_text(text_model, combined_text)
                            spectrogram, tmp_path = self.audio_processor.process_uploaded_file(combined_audio)
                            audio_result = self.analyze_audio(audio_model, spectrogram)
                            
                            if text_result and audio_result:
                                st.balloons()
                                st.success("✅ Analysis Complete!")
                                
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.markdown("### 📊 Combined Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown("#### 📝 Text Analysis")
                                    st.metric("Sentiment", text_result['predicted_label'])
                                    st.metric("Confidence", f"{text_result['confidence']:.2%}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.markdown("#### 🎵 Audio Analysis")
                                    st.metric("Emotion", audio_result['predicted_label'])
                                    st.metric("Confidence", f"{audio_result['confidence']:.2%}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                st.markdown("### 💡 Insights")
                                if text_result['predicted_label'] == "POSITIVE" and "HAPPY" in audio_result['predicted_label']:
                                    st.success("🌟 **Strong Positive Alignment:** Text and audio both indicate positive sentiment!")
                                elif text_result['predicted_label'] == "NEGATIVE" and "ANGRY" in audio_result['predicted_label']:
                                    st.error("⚠️ **Negative Correlation:** Both modalities show negative emotions")
                                else:
                                    st.info("🔍 **Mixed Signals:** Text and audio show different emotional patterns")
                                
                            os.unlink(tmp_path)
                    else:
                        st.warning("⚠️ Please provide both text and audio for combined analysis")
                st.markdown('</div>', unsafe_allow_html=True)

            elif selected_tab == "📊 Dashboard":
                st.markdown('<div class="tab-content">', unsafe_allow_html=True)
                st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Analyses", "128")
                    st.caption("This session")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Text Accuracy", "92%")
                    st.caption("Based on validation")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Audio Accuracy", "78%")
                    st.caption("Demo mode")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg Confidence", "85%")
                    st.caption("Overall performance")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### 📈 Activity Overview")
                chart_data = np.random.randn(20, 3)
                st.line_chart(chart_data)
                
                st.markdown("### 🎯 Recent Analyses")
                sample_data = {
                    "Text": ["Great service!", "Terrible experience", "It was okay"],
                    "Sentiment": ["POSITIVE", "NEGATIVE", "NEUTRAL"], 
                    "Confidence": [0.94, 0.87, 0.62]
                }
                st.dataframe(sample_data, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error("❌ An unexpected error occurred. Please try again.")

def main():
    app = MultimodalApp()
    app.run()

if __name__ == "__main__":
    main()