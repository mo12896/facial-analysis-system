from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from retinaface import RetinaFace

# from supervision.draw.color import ColorPalette
# from supervision.geometry.dataclasses import Point
# from supervision.tools.detections import BoxAnnotator, Detections
# from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
# from supervision.video.dataclasses import VideoInfo
# from supervision.video.sink import VideoSink
from utils.constants import OPENCV_MODEL


class FaceDetector(ABC):
    @abstractmethod
    def __init__(self, face_detector):
        """Base constructor for all face detectors.

        Args:
            face_detector: Face detector object.
        """
        self.face_detector = face_detector
        self.bboxes = []

    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        """Abstract method to detect faces in a given frame.

        Args:
            frame (np.ndarray): Current frame

        Returns:
            Tuple[list, np.ndarray]: Tuple of face crops and bounding boxes
        """

    def display_faces(self, frame: np.ndarray) -> np.ndarray:
        """Displays the detected faces in a given frame.

        Args:
            frame (np.ndarray): Current frame

        Returns:
            np.ndarray: Frame with detected faces
        """
        frame_cpy = np.copy(frame)
        for bbox in self.bboxes:
            cv2.rectangle(
                frame_cpy,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 0, 255),
                thickness=5,
            )
        cv2.imshow("Detected Faces", frame_cpy)
        return frame_cpy


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

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bboxes = self.face_detector.detectMultiScale(gray_frame)
        self.bboxes = np.array([(x, y, x + w, y + h) for (x, y, w, h) in bboxes])

        if self.bboxes.any():
            face_crops = [frame[y:h, x:w] for (x, y, w, h) in self.bboxes]
            return (
                face_crops,
                self.bboxes,
            )
        raise ValueError("No faces detected")


# TODO: Write own Detections class to hold the relevant data!
class RetinaFaceDetector(FaceDetector):
    """Face detector using RetinaFace."""

    def __init__(self, face_detector=RetinaFace):
        if len(tf.config.list_physical_devices("GPU")) < 1:
            raise ValueError("No GPU detected!")
        super().__init__(face_detector)

    def detect_faces(self, frame: np.ndarray) -> Tuple[list, np.ndarray]:
        faces = self.face_detector.detect_faces(frame)
        self.bboxes = np.array([faces[face]["facial_area"] for face in faces])

        if self.bboxes.any():
            face_crops = [frame[y:h, x:w] for (x, y, w, h) in self.bboxes]
            return (
                face_crops,
                self.bboxes,
            )
        raise ValueError("No faces detected")
