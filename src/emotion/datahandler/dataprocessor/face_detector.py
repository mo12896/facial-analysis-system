# import os
# import sys
from abc import ABC, abstractmethod

# grandparent_folder = os.path.abspath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
# )
# sys.path.append(grandparent_folder)
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace
from utils.annotator import BoxAnnotator
from utils.color import Color
from utils.constants import OPENCV_MODEL
from utils.detections import Detections
from utils.utils import timer

mp_face_detection = mp.solutions.face_detection


class FaceDetector(ABC):
    @abstractmethod
    def __init__(self, face_detector):
        """Base constructor for all face detectors.

        Args:
            face_detector: Face detector object.
        """
        self.face_detector = face_detector

    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> Detections:
        """Abstract method to detect faces in a given frame.

        Args:
            frame (np.ndarray): Current frame

        Returns:
            Detections: Object which holds the bounding boxes, confidences, and class ids
        """


def create_face_detector(detector: str) -> FaceDetector:
    """Factory method to create face detector objects.

    Args:
        detector (str): Name of the face detector

    Raises:
        ValueError: If the given detector is not supported!

    Returns:
        FaceDetector: Face detector object
    """
    if detector == "retinaface":
        return RetinaFaceDetector()
    elif detector == "opencv":
        face_detector = cv2.CascadeClassifier(str(OPENCV_MODEL))
        return OpenCVFaceDetector(face_detector)
    elif detector == "mediapipe":
        return MediaPipeFaceDetector()
    else:
        raise ValueError("The chosen face detector is not supported!")


class OpenCVFaceDetector(FaceDetector):
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, face_detector: cv2.CascadeClassifier):
        super().__init__(face_detector)

    @timer
    def detect_faces(self, frame: np.ndarray) -> Detections:
        # frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.detectMultiScale(frame)
        detections = Detections(
            bboxes=np.array(bboxes),
            confidence=np.ones(len(bboxes), dtype=np.float32),
            class_id=np.zeros(len(bboxes), dtype=np.int32),
        )

        if len(detections.bboxes) > 0:
            return detections
        raise ValueError("No faces detected")


class RetinaFaceDetector(FaceDetector):
    """Face detector using RetinaFace."""

    def __init__(self, face_detector=RetinaFace):
        if len(tf.config.list_physical_devices("GPU")) < 1:
            raise ValueError("No GPU detected!")
        super().__init__(face_detector)

    @timer
    def detect_faces(self, frame: np.ndarray) -> Detections:
        faces = self.face_detector.detect_faces(frame)
        detections = Detections.from_retinaface(faces)

        if len(detections.bboxes) > 0:
            return detections
        raise ValueError("No faces detected")


class MediaPipeFaceDetector(FaceDetector):
    def __init__(self, face_detector=mp_face_detection.FaceDetection):
        super().__init__(face_detector)

    @timer
    def detect_faces(self, frame: np.ndarray):
        face_detection = self.face_detector(
            model_selection=0, min_detection_confidence=0.5
        )
        # frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame)
        detections = Detections.from_mediapipe(results.detections, frame.shape[:2])

        if len(detections.bboxes) > 0:
            return detections
        raise ValueError("No faces detected")


if __name__ == "__main__":
    face_detector = create_face_detector("retinaface")
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)
    box_annotator = BoxAnnotator(color=Color.red())
    image = box_annotator.annotate(image, detections)
    plt.imshow(image)
    plt.show()
