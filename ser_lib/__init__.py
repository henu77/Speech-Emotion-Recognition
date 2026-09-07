"""ser_lib：语音情感识别库。"""

__version__ = "0.2.0"

from ser_lib.data import SERBatch, SERDataset, SERSample, TensorSpec
from ser_lib.engine import Trainer, TrainerConfig, evaluate
from ser_lib.inference import EmotionPredictor, PredictionResult
from ser_lib.models import CNNBaseline, ModelOutput, SERModel

__all__ = [
    "SERDataset", "SERSample", "SERBatch", "TensorSpec",
    "SERModel", "ModelOutput", "CNNBaseline",
    "Trainer", "TrainerConfig", "evaluate",
    "EmotionPredictor", "PredictionResult",
]
