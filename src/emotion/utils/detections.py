from typing import Optional, Union

import cv2
import numpy as np

from .color import Color, ColorPalette


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


class BoxAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette],
        thickness: int = 1,
        text_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
    ):

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding

    def annotate(
        self,
        frame: np.ndarray,
        detections: Detections,
        labels: Optional[list[str]] = None,
    ) -> np.ndarray:
        """Annotate frame with detections

        Args:
            frame (np.ndarray): Frame to annotate
            detections (Detections): Detections to annotate
            labels (list[str], optional): List of labels to annotate. Defaults to None.

        Returns:
            np.ndarray: Annotated frame
        """
        frame = frame.copy()
        for i, (bbox, confidence, class_id, _) in enumerate(detections):
            if isinstance(self.color, ColorPalette):
                color = self.color.by_idx(class_id)
            else:
                color = self.color

            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color.as_bgr(),
                self.thickness,
            )

            if labels is not None:
                text = f"Class '{labels[i]}': {confidence:.4f}"
            else:
                text = f"Class '{class_id}': {confidence:.4f}"

            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
            )[0]

            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (
                    int(bbox[0]) + text_size[0] + self.text_padding,
                    int(bbox[1]) - text_size[1] - self.text_padding,
                ),
                color.as_bgr(),
                -1,
            )

            frame = cv2.putText(
                frame,
                text,
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
            )

        return frame
