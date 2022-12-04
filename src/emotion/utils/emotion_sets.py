from dataclasses import dataclass
from typing import Union

import numpy as np

from .constants import (
    EKMAN_EMOTIONS,
    HUME_AI_EMOTIONS,
    NEUTRAL_EKMAN_EMOTIONS,
    THREE_EMOTIONS,
)


@dataclass(frozen=True, slots=True)
class EmotionSet:
    """
    Abstract dataclass to represent the structure for different emotions sets.
    """

    name: str
    emotion_count: int
    emotion_list: list[str]

    def get_emotions(self, indices: Union[int, np.ndarray]) -> np.ndarray:
        """Getter function for selecting emotions

        Args:
            indices (list): Indices by which emotions are selected.

        Returns:
            np.ndarray: Array of selected emotions.
        """
        assert np.logical_and(indices >= 0, indices < self.emotion_count).all()
        return np.array(self.emotion_list)[indices]


def generate_emotion_set(name: str) -> EmotionSet:
    """Abstract factory for creating different Emotion Sets.

    Args:
        name (str): Name of the desired Emotion Set

    Raises:
        ValueError: Raised if the name does not represent an Emotion Set
    Returns:
        EmotionSet: Final Emotion Set
    """
    if name == "three":
        return EmotionSet(name, 3, THREE_EMOTIONS)
    elif name == "ekman":
        return EmotionSet(
            name,
            6,
            EKMAN_EMOTIONS,
        )
    elif name == "neutral_ekman":
        return EmotionSet(
            name,
            7,
            NEUTRAL_EKMAN_EMOTIONS,
        )
    elif name == "hume_ai":
        return EmotionSet(
            name,
            27,
            HUME_AI_EMOTIONS,
        )
    else:
        raise ValueError(f'The chosen emotion set "{name}" does not exist!')
