from typing import Union

import cv2
import numpy as np

from .color import Color, ColorPalette
from .constants import PAIRS
from .detections import Detections


# TODO: can only be used, as soon as PoseEstimator is fixed!
class KeyPointAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette],
        thickness: int = 1,
        radius: int = 5,
    ):

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.radius: int = radius

    def annotate(
        self,
        image: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:
        """Annotate frame with detections

        Args:
            frame (np.ndarray): Frame to annotate
            detections (Detections): Detections to annotate

        Returns:
            np.ndarray: Annotated frame
        """
        image = image.copy()
        # Connect keypoints for each person
        for i, person_keypoints in enumerate(detections.keypoints):

            for pair in PAIRS:
                part_a = person_keypoints[pair[0]]
                part_b = person_keypoints[pair[1]]

                # Draw keypoints
                cv2.circle(
                    image,
                    (int(person_keypoints[i][0]), int(person_keypoints[i][1])),
                    self.radius,
                    self.color.as_bgr(),
                    self.thickness,
                )
                # Draw lines
                cv2.line(
                    image,
                    (int(part_a[0]), int(part_a[1])),
                    (int(part_b[0]), int(part_b[1])),
                    self.color.as_bgr(),
                    self.thickness,
                )

        return image
