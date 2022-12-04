from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import dlib
import numpy as np

from .face_detector import FaceDetector

# TODO: Implement BYOL tracker
# from yolox.tracker.byte_tracker import BYTETracker


class Tracker(ABC):
    """Base constructor for all trackers, using the Bridge pattern
    to decouple the tracker from the face detector."""

    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector

    @abstractmethod
    def track_faces(
        self, image: np.ndarray, frame_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Function to track faces in a given frame.

        Args:
            image (np.ndarray): Current frame
            frame_count (int): Index of the current frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: Face crops and bounding boxes
        """


class DlibTracker(Tracker):
    def __init__(self, face_detector: FaceDetector, detection_frequency: int = 10):
        super().__init__(face_detector)
        self.detection_frequency = detection_frequency

    def track_faces(
        self, image: np.ndarray, frame_count: int
    ) -> Tuple[list, np.ndarray]:
        # For frame_count >= 0, the detections become more accurate!
        if not frame_count or not frame_count % self.detection_frequency:
            self.trackers: list = []

            face_crops, bboxes = self.face_detector.detect_faces(image)
            # self.face_detector.display_faces(image)

            for (x, y, w, h) in bboxes:
                cv2.rectangle(
                    image,
                    (x, y),
                    (w, h),
                    (255, 0, 0),
                    thickness=2,
                )

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, w, h)

                tracker.start_track(image, rect)
                self.trackers.append(tracker)

            return (face_crops, np.array(bboxes))
        else:
            bboxes = []

            for track in self.trackers:
                track.update(image)
                pos = track.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                bboxes.append((startX, startY, endX, endY))

            face_crops = [image[y : y + h, x : x + w] for (x, y, w, h) in bboxes]

            return (face_crops, np.array(bboxes))
