from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass(frozen=True, slots=True)
class EmotionSet:
    """
    Abstract dataclass to represent the structure for different emotions sets.
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


class EmotionSetFactory:
    """
    Abstract factory for creating different Emotion Sets.
    """

    @staticmethod
    def generate(name: str) -> EmotionSet:
        """Generate a new Emotion Set

        Args:
            name (str): Name of the desired Emotion Set

        Raises:
            ValueError: Raised if the name does not represent an Emotion Set
        Returns:
            EmotionSet: Final Emotion Set
        """
        if name == "three":
            return EmotionSet(name, 3, ["positive", "negative", "neutral"])
        elif name == "ekman":
            return EmotionSet(
                name,
                6,
                ["anger", "surprise", "disgust", "enjoyment", "fear", "sadness"],
            )
        elif name == "neutral_ekman":
            return EmotionSet(
                name,
                7,
                emotion_list=[
                    "anger",
                    "surprise",
                    "disgust",
                    "enjoyment",
                    "fear",
                    "sadness",
                    "neutral",
                ],
            )
        elif name == "hume_ai":
            return EmotionSet(
                name,
                27,
                [
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
                ],
            )
        else:
            raise ValueError(f'The chosen emotion set "{name}" does not exist!')
