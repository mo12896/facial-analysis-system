from typing import Protocol, Tuple

import cv2
import numpy as np


class FaceDetector(Protocol):
    def __init__(self, face_detector):
        ...

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        ...


class OpenCVFaceDetector:
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, face_detector):
        self.face_detector = face_detector

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = self.face_detector.detectMultiScale(gray_frame)
        face_crops = [frame[y : y + h, x : x + w] for (x, y, w, h) in bboxes]
        return (
            face_crops,
            bboxes,
        )
