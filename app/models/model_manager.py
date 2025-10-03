from app.utils.logger import logger
from app.utils.config import config
from .audio_model import AudioModel
from .text_model import TextModel

class ModelManager:
    _instance = None
    _text_model = None
    _audio_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    @property
    def text_model(self):
        if self._text_model is None:
            logger.info("Loading text model...")
            self._text_model = TextModel()
        return self._text_model

    @property
    def audio_model(self):
        if self._audio_model is None:
            logger.info("Loading audio model...")
            self._audio_model = AudioModel(
                config.model.AUDIO_MODEL_PATH,
                config.model.NUM_AUDIO_CLASSES
            )
        return self._audio_model

    def get_models(self):
        return self.text_model, self.audio_model