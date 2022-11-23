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
        self.tracker = dlib.correlation_tracker()

        self.trackers: list = []

    def track_faces(
        self, image: np.ndarray, frame_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Has to be reinitialized every time a face gets excluded!
        if frame_count == 0:
            face_crops, bboxes = self.face_detector.detect_faces(image)

            for (x, y, w, h) in bboxes:
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    thickness=2,
                )

                rect = dlib.rectangle(x, y, x + w, y + h)

                self.tracker.start_track(image, rect)
                self.trackers.append(self.tracker)

            return (np.array(face_crops), bboxes)

        else:
            bboxes = []

            for tracker in self.trackers:
                tracker.update(image)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                bboxes.append((startX, startY, endX, endY))

            face_crops = [image[y : y + h, x : x + w] for (x, y, w, h) in bboxes]

            # TODO: Fix Broadcasting Error!!!
            return (np.array(face_crops), np.array(bboxes))
