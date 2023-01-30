from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import dlib
import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from utils.detections import Detections
from yolox.tracker.byte_tracker import BYTETracker, STrack

from .face_detector import FaceDetector


class Tracker(ABC):
    """Base constructor for all trackers, using the Bridge pattern
    to decouple the tracker from the face detector."""

    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector

    @abstractmethod
    def track_faces(self, image: np.ndarray) -> Detections:
        """Function to track faces in a given frame.

        Args:
            image (np.ndarray): Current frame
            frame_count (int): Index of the current frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: Face crops and bounding boxes
        """


def create_tracker(tracker: str, face_detector: FaceDetector, *args: Any) -> Tracker:
    """Factory method to create a tracker.

    Args:
        tracker (str): Name of the tracker
        face_detector (FaceDetector): Face detector

    Returns:
        Tracker: Tracker object
    """
    if tracker == "dlib":
        det_freq = [arg["DETECT_FREQ"] for arg in args if isinstance(arg, dict)][0]
        if det_freq:
            return DlibTracker(
                face_detector=face_detector, detection_frequency=det_freq
            )
        raise ValueError("Invalid detection frequency!")
    elif tracker == "byte":
        byte_args = [arg for arg in args if isinstance(arg, BYTETrackerArgs)][0]
        if byte_args:
            return ByteTracker(face_detector, byte_args)
        raise ValueError("Invalid args for BYTE tracker!")
    else:
        raise ValueError("Invalid tracker name!")


class DlibTracker(Tracker):
    """Dlib tracker implementation."""

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
    """Arguments for the BYTETracker from YOLOX."""

    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


class ByteTracker(Tracker):
    """Wrapper for the BYTETracker from YOLOX."""

    def __init__(
        self, face_detector: FaceDetector, args: BYTETrackerArgs = BYTETrackerArgs()
    ):
        super().__init__(face_detector)
        self.tracker = BYTETracker(args)

    def track_faces(self, image: np.ndarray, frame_count: int) -> Detections:
        detections = self.face_detector.detect_faces(image)
        tracks = self.tracker.update(
            output_results=self.detections2boxes(detections=detections),
            img_info=image.shape,
            img_size=image.shape,
        )

        tracker_id = self.match_detections_with_tracks(
            detections=detections, tracks=tracks
        )
        detections.tracker_id = np.array(tracker_id)
        return detections

    def match_detections_with_tracks(
        self, detections: Detections, tracks: list[STrack]
    ):
        if not np.any(detections.bboxes) or len(tracks) == 0:
            return np.empty((0,))

        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.bboxes)
        track2detection = np.argmax(iou, axis=1)

        tracker_ids = [None] * len(detections)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id

        return tracker_ids

    @staticmethod
    def detections2boxes(detections: Detections) -> np.ndarray:
        return np.hstack((detections.bboxes, detections.confidence[:, np.newaxis]))

    @staticmethod
    def tracks2boxes(tracks: list[STrack]) -> np.ndarray:
        return np.array([track.tlbr for track in tracks], dtype=float)
