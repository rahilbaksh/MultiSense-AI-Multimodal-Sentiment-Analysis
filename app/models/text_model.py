from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from app.utils.logger import logger
from app.utils.config import config

class TextModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                framework="pt"
            )
            logger.info(f"✅ Text model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading text model: {e}")
            try:
                self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    return_all_scores=True
                )
                logger.info(f"✅ Fallback to default model: {self.model_name}")
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                raise

    def predict(self, text: str):
        try:
            results = self.pipeline(text)
            return results[0]
        except Exception as e:
            logger.error(f"Error in text prediction: {e}")
            raise
