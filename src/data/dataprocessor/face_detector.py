from typing import Protocol, Tuple

import numpy as np
import cv2
from enum import Enum


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

    def detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def display_detected_faces(self, frame: np.ndarray):
        ...


class OpenCVFaceDetector:
    def __init__(
        self, face_detector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    ):
        self.face_detector = face_detector
        self.bboxes: np.ndarray = []

    def detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bboxes = self.face_detector.detectMultiScale(gray_frame)
        face_crops = [
            frame[y : y + h, x : x + w] for (x, y, w, h) in self.face_coordinates
        ]
        return (
            face_crops,
            self.bboxes,
        )

    def display_detected_faces(self, frame: np.ndarray) -> np.ndarray[float()]:
        for (x, y, w, h) in self.bboxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
