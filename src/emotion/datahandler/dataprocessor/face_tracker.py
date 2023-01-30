from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import dlib
import numpy as np
from utils.detections import Detections

from .face_detector import FaceDetector

# from onemetric.cv.utils.iou import box_iou_batch
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
        self.confidence = None

    def track_faces(self, image: np.ndarray, frame_count: int) -> Detections:
        # For frame_count >= 0, the detections become more accurate!
        if not frame_count or not frame_count % self.detection_frequency:
            self.trackers: list = []

            detections = self.face_detector.detect_faces(image)
            self.confidence = detections.confidence

            for (x, y, w, h) in detections.bboxes:

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, w, h)

                tracker.start_track(image, rect)
                self.trackers.append(tracker)

            return detections

        bboxes = []

        for track in self.trackers:
            track.update(image)
            pos = track.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            bboxes.append((startX, startY, endX, endY))

        detections = Detections(
            bboxes=np.array(bboxes),
            confidence=self.confidence,
            class_id=np.zeros(len(bboxes), dtype=np.int32),
        )

        return detections


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
