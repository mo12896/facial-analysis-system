from abc import ABC, abstractmethod
from dataclasses import dataclass

# import dlib
import numpy as np
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.byte_tracker import BYTETracker, STrack

from src.emotion.features.detections import Detections
from src.emotion.utils.utils import timer


class Tracker(ABC):
    """Abstract class for face trackers."""

    @abstractmethod
    def __init__(self, parameters: dict = {}):
        """Constructor for the Tracker class."""

    @abstractmethod
    def track_faces(self, detections: Detections, image: np.ndarray) -> Detections:
        """Function to track faces in a given frame.

        Args:
            image (np.ndarray): Current frame
            frame_count (int): Index of the current frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: Face crops and bounding boxes
        """


def create_tracker(paramaters: dict = {}):
    """Factory method to create a tracker.

    Args:
        tracker (str): Name of the tracker
        face_detector (FaceDetector): Face detector

    Returns:
        Tracker: Tracker object
    """
    if paramaters["type"] == "dlib":
        raise NotImplementedError("Dlib tracker not implemented!")
    elif paramaters["type"] == "byte":
        return ByteTracker(paramaters)
    else:
        raise ValueError("Invalid tracker name!")


# # TODO: Not fully implemented
# class DlibTracker(Tracker):
#     """Dlib tracker implementation."""

#     def __init__(self, parameters: dict = {}):
#         self.detection_frequency = parameters.get("detection_frequency", 10)
#         self.confidence = None
#         self.first_run = True

#     def track_faces(self, detections: Detections, image: np.ndarray) -> Detections:
#         # For frame_count >= 0, the detections become more accurate!
#         if self.first_run:
#             self.trackers: list = []

#             self.confidence = detections.confidence

#             for (x, y, w, h) in detections.bboxes:

#                 tracker = dlib.correlation_tracker()
#                 rect = dlib.rectangle(x, y, w, h)

#                 tracker.start_track(image, rect)
#                 self.trackers.append(tracker)

#             self.first_run = False

#             return detections

#         bboxes = []

#         for track in self.trackers:
#             track.update(image)
#             pos = track.get_position()

#             startX = int(pos.left())
#             startY = int(pos.top())
#             endX = int(pos.right())
#             endY = int(pos.bottom())

#             bboxes.append((startX, startY, endX, endY))

#         detections = Detections(
#             bboxes=np.array(bboxes),
#             confidence=self.confidence,
#             class_id=np.zeros(len(bboxes), dtype=np.int32),
#         )

#         return detections


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

    def __init__(self, parameters: dict = {}):
        args = parameters.get("args", BYTETrackerArgs)
        self.tracker = BYTETracker(args)
        self.person_ids: list = []

    @timer
    def track_faces(self, detections: Detections, image: np.ndarray) -> Detections:
        """Track faces in a given frame.

        Args:
            detections (Detections): Detections to track
            image (np.ndarray): The current frame

        Returns:
            Detections: Detections with tracker ids
        """
        tracks = self.tracker.update(
            output_results=self.detections2boxes(detections=detections),
            img_info=image.shape,
            img_size=image.shape,
        )

        detections.tracks = np.array([track.tlbr for track in tracks])

        tracker_id = self.match_detections_with_tracks(
            detections=detections, tracks=tracks
        )
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array(
            [tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool
        )
        detections.filter(mask=mask, inplace=True)

        return detections

    def match_detections_with_tracks(
        self, detections: Detections, tracks: list[STrack]
    ) -> list:
        """Match tracks interface with detections interface.

        Args:
            detections (Detections): Detections object
            tracks (list[STrack]): List of tracks to match

        Returns:
            list: List of track ids
        """
        if not np.any(detections.bboxes) or len(tracks) == 0:
            return np.empty((0,))

        tracks_boxes = self.tracks2boxes(tracks=tracks)
        iou = box_iou_batch(tracks_boxes, detections.bboxes)
        track2detection = np.argmax(iou, axis=1)

        tracker_ids = [None] * len(detections)

        for tracker_index, detection_index in enumerate(track2detection):
            if iou[tracker_index, detection_index] != 0:
                tracker_ids[detection_index] = tracks[tracker_index].track_id
                # tracker_ids[detection_index] = self.person_ids[tracker_index]

        return tracker_ids

    @staticmethod
    def detections2boxes(detections: Detections) -> np.ndarray:
        """Get detections in the format of BYTETracker."""
        return np.hstack((detections.bboxes, detections.confidence[:, np.newaxis]))

    @staticmethod
    def tracks2boxes(tracks: list[STrack]) -> np.ndarray:
        """Get tracks in the format of BYTETracker."""
        return np.array([track.tlbr for track in tracks], dtype=float)
