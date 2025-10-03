import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    TEXT_MODEL_NAME: str = "distilbert-base-uncased"
    AUDIO_MODEL_PATH: str = "models/audio_model.pth"
    NUM_AUDIO_CLASSES: int = 4
    SAMPLE_RATE: int = 22050
    DURATION: int = 4
    N_MELS: int = 128
    MAX_LENGTH: int = 512

@dataclass
class AppConfig:
    HOST: str = "0.0.0.0"
    PORT: int = 8501
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.app = AppConfig()

config = Config()