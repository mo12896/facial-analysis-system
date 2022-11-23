from enum import Enum


class DeepFaceBackends(Enum):
    """Enum for the different backends."""

    OPENCV = "opencv"
    DLIB = "dlib"
    SSD = "ssd"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    MEDIAPIPE = "mediapipe"
