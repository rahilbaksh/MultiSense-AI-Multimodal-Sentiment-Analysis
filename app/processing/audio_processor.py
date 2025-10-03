import librosa
import numpy as np
import tempfile
import soundfile as sf
from app.utils.logger import logger
from app.utils.config import config

class AudioProcessor:
    def __init__(self):
        self.sample_rate = config.model.SAMPLE_RATE
        self.duration = config.model.DURATION
        self.n_mels = config.model.N_MELS

    def load_audio(self, file_path: str):
        try:
            audio, sr = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                duration=self.duration
            )
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise

    def create_spectrogram(self, audio: np.ndarray):
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=2048,
                hop_length=512
            )
            
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
            
            return log_mel_spec
            
        except Exception as e:
            logger.error(f"Error creating spectrogram: {e}")
            raise

    def process_uploaded_file(self, uploaded_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            audio, _ = self.load_audio(tmp_path)
            spectrogram = self.create_spectrogram(audio)
            
            return spectrogram, tmp_path
            
        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}")
            raise