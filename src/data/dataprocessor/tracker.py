from typing import Protocol
from deepface import DeepFace
import cv2
from enum import Enum


class DeepFaceBackends(Enum):
    OPENCV = "opencv"
    DLIB = "dlib"
    SSD = "ssd"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"


class FaceDetector(Protocol):
    def __init__(self, face_detector):
        ...

    def detect_faces(self, image):
        ...

    def display_detected_faces(self, image):
        ...


class DeepFaceDetector:
    def __init__(self, face_detector: DeepFaceBackends):
        self.face_detector = face_detector

    def detect_faces(self, image):
        self.faces = self.face_detector.detectFace(
            image, detector_backend=self.face_detector
        )
        return self.faces

    def display_detected_faces(self, image):
        for face in self.faces:
            bounding_box = face["region"]
            cv2.rectangle(
                image,
                (bounding_box["x"], bounding_box["y"]),
                (
                    bounding_box["x"] + bounding_box["w"],
                    bounding_box["y"] + bounding_box["h"],
                ),
                (255, 0, 0),
                2,
            )
        return image
