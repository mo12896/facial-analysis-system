from abc import ABC
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class EmotionSet(ABC):
    """
    Abstract class to represent the datastructure for different emotions sets.
    """

    name: str
    emotion_count: int
    emotion_list: list(str)

    def get_emotions(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        """Getter function for selecting emotions

        Args:
            indices (list): Indices by which emotions are selected.

        Returns:
            np.ndarray: Array of selected emotions.
        """
        assert np.logical_and(indices >= 0, indices < self.emotion_count).all()
        return np.array(self.emotion_list)[indices]


@dataclass
class ThreeEmotions(EmotionSet):
    name = "three"
    emotion_count = 3
    emotion_list = ["positive", "negative", "neutral"]


@dataclass
class EkmanEmotions(EmotionSet):
    name = "ekman"
    emotion_count = 6
    emotion_list = ["anger", "surprise", "disgust", "enjoyment", "fear", "sadness"]


@dataclass
class NeutralEkmanEmotions(EmotionSet):
    name = "neutral_ekman"
    emotion_count = 7
    emotion_list = [
        "anger",
        "surprise",
        "disgust",
        "enjoyment",
        "fear",
        "sadness",
        "neutral",
    ]


@dataclass
class HumeAIEmotions(EmotionSet):
    name = "hume_ai"
    emotion_count = 27
    emotion_list = [
        "admiration",
        "adoration",
        "aesthetic appreciation",
        "amusement",
        "anger",
        "anxiety",
        "awe",
        "awkwardness",
        "neutral",
        "bordeom",
        "calmness",
        "confusion",
        "contempt",
        "craving",
        "disappointment",
        "disgust",
        "admiration",
        "adoration",
        "empathetic pain",
        "entrancement",
        "envy",
        "excitement",
        "fear",
        "guilt",
        "horror",
        "interest",
        "joy",
        "nostalgia",
        "pride",
        "relief",
        "romance",
        "sadness",
        "satisfaction",
        "secual desire",
        "surprise",
        "sympathy",
        "triumph",
    ]
