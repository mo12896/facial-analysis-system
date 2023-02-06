from typing import Optional

import numpy as np


class Detections:
    def __init__(
        self,
        bboxes: np.ndarray,
        confidence: np.ndarray,
        class_id: np.ndarray,
        emotion: Optional[np.ndarray] = None,
        tracker_id: Optional[np.ndarray] = None,
    ):
        self.bboxes = bboxes
        self.confidence = confidence
        self.class_id = class_id
        self.emotion = emotion
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
                self.emotion[i] if self.emotion is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
            )

    @classmethod
    def from_retinaface(cls, retinaface_output: dict):
        """Create Detections object from RetinaFace output"""

        bboxes = []
        confidence = []
        class_id = []
        emotions = []

        for face in retinaface_output:
            bboxes.append(retinaface_output[face]["facial_area"])
            confidence.append(retinaface_output[face]["score"])
            class_id.append(face)

        # Note that setting the dtype for class_id is important to keep the final output strings!
        return cls(
            np.array(bboxes),
            np.array(confidence),
            np.array(class_id, dtype="U10"),
            np.array(emotions),
        )

    @classmethod
    def from_mediapipe(cls, mediapipe_output: dict, image_size):
        """Create Detections object from MediaPipe output"""

        bboxes = []
        confidence = []
        class_id = []
        image_height, image_width = image_size
        lines = str(mediapipe_output).strip().split(",")

        # Bulky parser, because mediapipe...
        for line in lines:
            parts = line.strip().split("\n")
            label_id = int(parts[0].split(":")[1].strip())
            score = float(parts[1].split(":")[1].strip())
            xmin = float(parts[5].split(":")[1].strip())
            ymin = float(parts[6].split(":")[1].strip())
            width = float(parts[7].split(":")[1].strip())
            height = float(parts[8].strip().split(":")[1].strip())

            xmin = int(xmin * image_width)
            ymin = int(ymin * image_height)
            width = int(width * image_width)
            height = int(height * image_height)

            bbox = np.array([xmin, ymin, xmin + width, ymin + height])

            bboxes.append(bbox)
            confidence.append(score)
            class_id.append(label_id)

        # Note that setting the dtype for class_id is important to keep the final output strings!
        return cls(
            np.array(bboxes), np.array(confidence), np.array(class_id, dtype="U10")
        )

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
