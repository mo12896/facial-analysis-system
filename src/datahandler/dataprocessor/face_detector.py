from enum import Enum
from typing import Protocol, Tuple

import cv2
import numpy as np


class DeepFaceBackends(Enum):
    """Enum for the different backends."""

    OPENCV = "opencv"
    DLIB = "dlib"
    SSD = "ssd"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"


class FaceDetector(Protocol):
    def __init__(self, face_detector):
        ...

    def detect_faces(self, frame: np.ndarray) -> Tuple[list[np.ndarray], np.ndarray]:
        ...

    def display_detected_faces(self, frame: np.ndarray) -> np.ndarray:
        ...


class OpenCVFaceDetector:
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.bboxes: np.ndarray = np.array([])

    def detect_faces(self, frame: np.ndarray) -> Tuple[list[np.ndarray], np.ndarray]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bboxes = self.face_detector.detectMultiScale(gray_frame)
        face_crops = [frame[y : y + h, x : x + w] for (x, y, w, h) in self.bboxes]
        return (
            face_crops,
            self.bboxes,
        )

    def display_detected_faces(self, frame: np.ndarray) -> np.ndarray:
        for (x, y, w, h) in self.bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
