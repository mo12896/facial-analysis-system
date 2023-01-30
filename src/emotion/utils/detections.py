from typing import Optional

import numpy as np


class Detections:
    def __init__(
        self,
        xyxy: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        tracker_id: Optional[np.ndarray] = None,
    ):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id

        n = len(self.xyxy)
        validators = [
            (isinstance(self.xyxy, np.ndarray) and self.xyxy.shape == (n, 4)),
            (isinstance(self.confidence, np.ndarray) and self.confidence.shape == (n,)),
            (isinstance(self.class_id, np.ndarray) and self.class_id.shape == (n,)),
            self.tracker_id is None
            or (
                isinstance(self.tracker_id, np.ndarray)
                and self.tracker_id.shape == (n,)
            ),
        ]

        if not all(validators):
            raise ValueError(
                "xyxy must be 2d np.ndarray with (n, 4) shape, "
                "confidence must be 1d np.ndarray with (n,) shape, "
                "class_id must be 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def __len__(self):
        return len(self.xyxy)

    def __iter__(self):
        """Iterate over detections and yield a tuple of (xyxy, confidence, class_id, tracker_id)"""
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_retinaface(cls, retinaface_output: np.ndarray):
        """Create Detections object from RetinaFace output"""
        xyxy = np.ndarray([face["facial_area"] for face in retinaface_output])
        confidence = np.ndarray([face["score"] for face in retinaface_output])
        class_id = np.zeros(len(xyxy), dtype=np.int32)
        return cls(xyxy, confidence, class_id)

    @classmethod
    def from_opencv(cls, opencv_output: np.ndarray):
        """Create Detections object from OpenCV output"""
        xyxy = np.ndarray(opencv_output)
        confidence = np.ones(len(xyxy), dtype=np.float32)
        class_id = np.zeros(len(xyxy), dtype=np.int32)
        return cls(xyxy, confidence, class_id)

    def filter(self, mask: np.ndarray, inplace: bool = False):
        """Filter detections by mask

        Args:
            mask (np.ndarray): Mask of shape (n,) containing boolean values to filter detections
            inplace (bool, optional): If True, filter detections inplace. Defaults to False.

        """
        if inplace:
            self.xyxy = self.xyxy[mask]
            self.confidence = self.confidence[mask]
            self.class_id = self.class_id[mask]
            self.tracker_id = (
                self.tracker_id[mask] if self.tracker_id is not None else None
            )
            return self
        else:
            return Detections(
                xyxy=self.xyxy[mask],
                confidence=self.confidence[mask],
                class_id=self.class_id[mask],
                tracker_id=self.tracker_id[mask]
                if self.tracker_id is not None
                else None,
            )
