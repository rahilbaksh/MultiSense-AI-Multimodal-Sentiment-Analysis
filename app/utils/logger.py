import logging
import sys
from .config import config

def setup_logger():
    logger = logging.getLogger("multimodal-sentiment")
    logger.setLevel(getattr(logging, config.app.LOG_LEVEL))
    
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

logger = setup_logger()