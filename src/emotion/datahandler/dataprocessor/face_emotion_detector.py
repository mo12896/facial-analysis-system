from abc import ABC, abstractmethod

import numpy as np
from deepface import DeepFace
from rmn import RMN

from src.emotion.utils.detections import Detections


class EmotionDetector(ABC):
    @abstractmethod
    def __init__(self, parameters: dict = {}) -> None:
        """Constructor for the EmotionDetector class.

        Args:
            parameters (dict, optional): Dictionary containing the parameters for the
            emotion detector. Defaults to {}.
        """
        self.parameters = parameters

    @abstractmethod
    def detect_emotions(self, detections: Detections, image: np.ndarray) -> Detections:
        """Detect emotions for a set of detections.

        Args:
            detections (Detections): Detection object.
            image (np.ndarray): The current frame of the video.

        Returns:
            Detections: Detection object with the emotions.
        """


def create_emotion_detector(parameters: dict = {}) -> EmotionDetector:
    """Factory method for creating an emotion detector.

    Args:
        parameters (dict, optional): Dictionary containing the parameters for the
        emotion detector. Defaults to {}.

    Raises:
        ValueError: Raised if the emotion detector is not supported.

    Returns:
        EmotionDetector: Emotion detector object.
    """
    if parameters["type"] == "deepface":
        return DeepFaceEmotionDetector(parameters)
    elif parameters["type"] == "rmn":
        return RMNEmotionDetector(parameters)
    else:
        raise ValueError(f"Emotion detector {parameters['type']} is not supported")


class DeepFaceEmotionDetector(EmotionDetector):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        self.emotion_detector = DeepFace

    def detect_emotions(self, detections: Detections, image: np.ndarray) -> Detections:
        for bbox in detections.bboxes:
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            analyze = self.emotion_detector.analyze(img, actions=["emotion"])
            emotions = analyze["emotion"]
            detections.emotions.append(emotions)
        return detections


class RMNEmotionDetector(EmotionDetector):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        self.emotion_detector = RMN()

    def detect_emotions(self, detections: Detections, image: np.ndarray) -> Detections:
        for bbox in detections.bboxes:
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            emotions = self.emotion_detector.detect_emotion_for_single_frame(img)
            detections.emotions.append(emotions)
        return detections
