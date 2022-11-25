from typing import Protocol, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from retinaface import RetinaFace

OPENCV_MODEL = "/home/moritz/anaconda3/envs/emotion/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml"


class FaceDetectorFactory:
    @staticmethod
    def create_face_detector(detector: str) -> "FaceDetector":
        if detector == "retinaface":
            return RetinaFaceDetector()
        elif detector == "opencv":
            face_detector = cv2.CascadeClassifier(OPENCV_MODEL)
            return OpenCVFaceDetector(face_detector)
        else:
            raise ValueError("Invalid face detector")


class FaceDetector(Protocol):
    def __init__(self, face_detector):
        ...

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        ...

    def display_faces(self, frame: np.ndarray) -> np.ndarray:
        ...


class OpenCVFaceDetector:
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, face_detector: cv2.CascadeClassifier):
        self.face_detector = face_detector
        self.bboxes = []

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bboxes = self.face_detector.detectMultiScale(gray_frame)
        if self.bboxes.any():
            face_crops = [frame[y : y + h, x : x + w] for (x, y, w, h) in self.bboxes]
            return (
                face_crops,
                self.bboxes,
            )
        else:
            raise ValueError("No faces detected")

    def display_faces(self, frame: np.ndarray) -> np.ndarray:
        frame_cpy = np.copy(frame)
        for bbox in self.bboxes:
            cv2.rectangle(
                frame_cpy,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                (0, 0, 255),
                thickness=1,
            )
        cv2.imshow("Detected Faces", frame_cpy)
        return frame_cpy


class RetinaFaceDetector:
    """Face detector using RetinaFace."""

    def __init__(self, face_detector=RetinaFace):
        self.face_detector = face_detector
        self.bboxes = []

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        faces = self.face_detector.detect_faces(frame)
        self.bboxes = np.array([faces[face]["facial_area"] for face in faces])

        if self.bboxes.any():
            face_crops = [frame[y : y + h, x : x + w] for (x, y, w, h) in self.bboxes]
            return (
                face_crops,
                self.bboxes,
            )
        else:
            raise ValueError("No faces detected")

    def display_faces(self, frame: np.ndarray) -> np.ndarray:
        frame_cpy = np.copy(frame)
        for bbox in self.bboxes:
            cv2.rectangle(
                frame_cpy,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 0, 255),
                5,
            )
        cv2.imshow("Detected Faces", frame_cpy)
        return frame_cpy
