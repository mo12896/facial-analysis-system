from typing import Protocol, Tuple

import cv2
import dlib
import numpy as np

from .face_detector import FaceDetector


class Tracker(Protocol):
    def __init__(self, face_detector: FaceDetector):
        ...

    def track_faces(
        self, image: np.ndarray, frame_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class DlibTracker:
    def __init__(self, face_detector: FaceDetector):
        self.face_detector = face_detector

        self.trackers: list = []

    def track_faces(
        self, image: np.ndarray, frame_count: int
    ) -> Tuple[list, np.ndarray]:
        # TODO: Has to be reinitialized every time a face gets excluded!
        if frame_count == 0:
            face_crops, bboxes = self.face_detector.detect_faces(image)
            self.face_detector.display_faces(image)

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
