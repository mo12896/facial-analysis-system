from abc import ABC, abstractmethod

import cv2
import numpy as np
from deepface.extendedmodels import Emotion
from rmn import RMN

from src.emotion.utils.detections import Detections
from src.emotion.utils.utils import timer


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
        self.emotion_detector = Emotion.loadModel()
        self.emo_label = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

    @timer
    def detect_emotions(self, detections: Detections, image: np.ndarray) -> Detections:

        emotions = []

        for bbox in detections.bboxes:
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)

            emotion_predictions = self.emotion_detector.predict(img_gray, verbose=0)[
                0, :
            ]

            sum_of_predictions = emotion_predictions.sum()

            emotions_dict = {}

            for i, emotion_label in enumerate(self.emo_label):
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                emotions_dict[emotion_label] = emotion_prediction

            emotions.append(emotions_dict)

        detections.emotion = np.array(emotions)

        return detections


# Faster! Not necessarily more accurate!
class RMNEmotionDetector(EmotionDetector):
    def __init__(self, parameters: dict = {}) -> None:
        super().__init__(parameters)
        self.emotion_detector = RMN(face_detector=False)

    @timer
    def detect_emotions(self, detections: Detections, image: np.ndarray) -> Detections:

        emotions = []

        for bbox in detections.bboxes:
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            result = self.emotion_detector.detect_emotion_for_single_face_image(img)

            if not result:
                emotions.append({})
                continue

            curr_dict = {}
            result = result[2]
            for d in result:
                curr_dict.update(d)

            emotions.append(curr_dict)

        detections.emotion = np.array(emotions)

        return detections
