# import os
# import sys
from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt

# import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis

from src.emotion.features.annotators.annotator import BoxAnnotator
from src.emotion.features.detections import Detections
from src.emotion.utils.color import Color
from src.emotion.utils.constants import OPENCV_MODEL
from src.emotion.utils.utils import timer

# from retinaface import RetinaFace


# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


# mp_face_detection = mp.solutions.face_detection


class FaceDetector(ABC):
    @abstractmethod
    def __init__(self, parameters: dict = {}):
        """Base constructor for all face detectors.

        Args:
            face_detector: Face detector object.
        """
        self.parameters = parameters

    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> Detections:
        """Abstract method to detect faces in a given frame.

        Args:
            frame (np.ndarray): Current frame

        Returns:
            Detections: Object which holds the bounding boxes, confidences, and class ids
        """


def create_face_detector(parameters: dict) -> FaceDetector:
    """Factory method to create face detector objects.

    Args:
        detector (str): Name of the face detector

    Raises:
        ValueError: If the given detector is not supported!

    Returns:
        FaceDetector: Face detector object
    """
    if parameters["type"] == "retinaface":
        raise NotImplementedError("RetinaFace is not supported!")
    elif parameters["type"] == "scrfd":
        return SCRFDFaceDetector(parameters)
    elif parameters["type"] == "opencv":
        return OpenCVFaceDetector(parameters)
    elif parameters["type"] == "mediapipe":
        raise NotImplementedError("MediaPipe is not supported!")
    else:
        raise ValueError("The chosen face detector is not supported!")


class OpenCVFaceDetector(FaceDetector):
    """Face detector using OpenCV's Haar Cascade Classifier."""

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters)
        self.face_detector = cv2.CascadeClassifier(str(OPENCV_MODEL))

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


# Legacy implementation from retinface package (15x slower than SCRFD)
# class RetinaFaceDetector(FaceDetector):
#     """Face detector using RetinaFace."""

#     def __init__(self, parameters: dict = {}):
#         super().__init__(parameters)
#         self.face_detector = RetinaFace

#     @timer
#     def detect_faces(self, frame: np.ndarray) -> Detections:
#         faces = self.face_detector.detect_faces(frame)
#         detections = Detections.from_retinaface(faces)

#         if len(detections.bboxes) > 0:
#             return detections
#         raise ValueError("No faces detected")


class SCRFDFaceDetector(FaceDetector):
    """SCRFD Face Detector using InsightFace's implementation."""

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters)
        self.face_detector = FaceAnalysis(allowed_modules=["detection"])
        self.face_detector.prepare(ctx_id=0, det_size=(640, 640))

    @timer
    def detect_faces(self, frame: np.ndarray) -> Detections:

        faces = self.face_detector.get(frame)
        detections = Detections.from_scrfd(faces)

        if len(detections.bboxes) > 0:
            return detections
        raise ValueError("No faces detected")


# class MediaPipeFaceDetector(FaceDetector):
#     def __init__(self, parameters: dict = {}):
#         super().__init__(parameters)
#         self.face_detector = mp_face_detection.FaceDetection

#     @timer
#     def detect_faces(self, frame: np.ndarray) -> Detections:
#         face_detection = self.face_detector(
#             model_selection=0, min_detection_confidence=0.5
#         )
#         # frame.flags.writeable = False
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detection.process(frame)
#         detections = Detections.from_mediapipe(results.detections, frame.shape[:2])

#         if len(detections.bboxes) > 0:
#             return detections
#         raise ValueError("No faces detected")


if __name__ == "__main__":
    # Uncomment the sys path for testing!
    face_detector = create_face_detector({"type": "scrfd"})
    image = cv2.imread("/home/moritz/Workspace/masterthesis/data/test_image.png")
    detections = face_detector.detect_faces(image)
    box_annotator = BoxAnnotator(color=Color.red())
    image = box_annotator.annotate(image, detections)
    plt.imshow(image)
    plt.show()
