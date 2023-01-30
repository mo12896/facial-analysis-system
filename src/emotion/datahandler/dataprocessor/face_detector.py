from abc import ABC, abstractmethod

import cv2
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace

# from supervision.geometry.dataclasses import Point
from utils.constants import OPENCV_MODEL
from utils.detections import Detections


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
        face_detector = cv2.CascadeClassifier(OPENCV_MODEL)
        return OpenCVFaceDetector(face_detector)
    else:
        raise ValueError("The chosen face detector is not supported!")


class OpenCVFaceDetector(FaceDetector):
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, face_detector: cv2.CascadeClassifier):
        super().__init__(face_detector)

    def detect_faces(self, frame: np.ndarray) -> Detections:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = self.face_detector.detectMultiScale(gray_frame)
        detections = Detections(
            bboxes=np.array(bboxes),
            confidence=np.ones(len(bboxes), dtype=np.float32),
            class_id=np.zeros(len(bboxes), dtype=np.int32),
        )

        if len(detections) > 0:
            return detections
        raise ValueError("No faces detected")


class RetinaFaceDetector(FaceDetector):
    """Face detector using RetinaFace."""

    def __init__(self, face_detector=RetinaFace):
        if len(tf.config.list_physical_devices("GPU")) < 1:
            raise ValueError("No GPU detected!")
        super().__init__(face_detector)

    def detect_faces(self, frame: np.ndarray) -> Detections:
        faces = self.face_detector.detect_faces(frame)
        detections = Detections.from_retinaface(faces)

        if len(detections) > 0:
            return detections
        raise ValueError("No faces detected")
