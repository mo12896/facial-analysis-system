from typing import Optional, Union

import cv2
import numpy as np

from .color import Color, ColorPalette
from .detections import Detections


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
