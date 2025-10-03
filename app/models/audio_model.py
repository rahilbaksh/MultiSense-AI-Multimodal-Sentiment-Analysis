import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from app.utils.logger import logger

class AudioCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 11, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class AudioModel:
    def __init__(self, model_path: str, num_classes: int = 4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioCNN(num_classes=num_classes)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Audio model loaded on device: {self.device}")
        
    def load_model(self, model_path: str):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Audio model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading audio model: {e}")
            logger.warning("Continuing in demo mode with random predictions")

    def predict(self, spectrogram):
        logger.warning("Using DEMO MODE - returning random predictions")
        random_predictions = np.random.dirichlet(np.ones(4), size=1)[0]
        return random_predictions
