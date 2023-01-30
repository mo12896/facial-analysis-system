from typing import Optional

import numpy as np


class Detections:
    def __init__(
        self,
        bboxes: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        tracker_id: Optional[np.ndarray] = None,
    ):
        self.bboxes = bboxes
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id

        n = len(self.bboxes)
        validators = [
            (isinstance(self.bboxes, np.ndarray) and self.bboxes.shape == (n, 4)),
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
                "bboxes must be 2d np.ndarray with (n, 4) shape, "
                "confidence must be 1d np.ndarray with (n,) shape, "
                "class_id must be 1d np.ndarray with (n,) shape, "
                "tracker_id must be None or 1d np.ndarray with (n,) shape"
            )

    def __len__(self):
        return len(self.bboxes)

    def __iter__(self):
        """Iterate over detections and yield a tuple of (bboxes, confidence, class_id, tracker_id)"""
        for i in range(len(self.bboxes)):
            yield (
                self.bboxes[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_retinaface(cls, retinaface_output: dict):
        """Create Detections object from RetinaFace output"""
        bboxes = np.array(
            [retinaface_output[face]["facial_area"] for face in retinaface_output]
        )
        confidence = np.array(
            [retinaface_output[face]["score"] for face in retinaface_output]
        )
        class_id = np.zeros(len(bboxes), dtype=np.int32)
        return cls(bboxes, confidence, class_id)

    @classmethod
    def from_array(cls, opencv_output: np.ndarray):
        """Create Detections object from OpenCV output"""
        bboxes = np.array(opencv_output)
        confidence = np.ones(len(bboxes), dtype=np.float32)
        class_id = np.zeros(len(bboxes), dtype=np.int32)
        return cls(bboxes, confidence, class_id)

    def filter(self, mask: np.ndarray, inplace: bool = False):
        """Filter detections by mask

        Args:
            mask (np.ndarray): Mask of shape (n,) containing boolean values to filter detections
            inplace (bool, optional): If True, filter detections inplace. Defaults to False.

        """
        if inplace:
            self.bboxes = self.bboxes[mask]
            self.confidence = self.confidence[mask]
            self.class_id = self.class_id[mask]
            self.tracker_id = (
                self.tracker_id[mask] if self.tracker_id is not None else None
            )
            return self
        else:
            return Detections(
                bboxes=self.bboxes[mask],
                confidence=self.confidence[mask],
                class_id=self.class_id[mask],
                tracker_id=self.tracker_id[mask]
                if self.tracker_id is not None
                else None,
            )
