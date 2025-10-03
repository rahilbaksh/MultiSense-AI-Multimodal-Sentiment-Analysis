import re
from app.utils.logger import logger

class TextProcessor:
    @staticmethod
    def clean_text(text: str):
        try:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text.lower()
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise

    @staticmethod
    def validate_text(text: str, min_length: int = 5):
        if not text or len(text.strip()) < min_length:
            raise ValueError(f"Text must be at least {min_length} characters long")
        return True