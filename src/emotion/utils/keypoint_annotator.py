import math
from typing import Union

import cv2
import numpy as np

from .color import Color, ColorPalette
from .constants import PAIRS, limb_seq, pose_colors
from .detections import Detections


class KeyPointAnnotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette],
        thickness: int = -1,
        radius: int = 4,
        stickwidth: int = 4,
    ):

        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.radius: int = radius
        self.stickwidth = stickwidth

    def annotate(
        self,
        image: np.ndarray,
        detections: Detections,
    ) -> np.ndarray:

        for i in range(18):
            for n in range(len(detections.body_pose_keypoints)):
                index = detections.body_pose_keypoints[n][i]
                if -1 in index:
                    continue
                x, y = index[0], index[1]
                cv2.circle(
                    image,
                    (int(x), int(y)),
                    self.radius,
                    pose_colors[i],
                    thickness=self.thickness,
                )

        for i in range(17):
            for n in range(len(detections.body_pose_keypoints)):
                index = detections.body_pose_keypoints[n][np.array(limb_seq[i]) - 1]
                if -1 in index:
                    continue
                cur_canvas = image.copy()
                Y = index.astype(int)[:, 0]
                X = index.astype(int)[:, 1]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly(
                    (int(mY), int(mX)),
                    (int(length / 2), self.stickwidth),
                    int(angle),
                    0,
                    360,
                    1,
                )
                cv2.fillConvexPoly(cur_canvas, polygon, pose_colors[i])
                image = cv2.addWeighted(image, 0.4, cur_canvas, 0.6, 0)

        return image

    # TODO: Still has to be debugged
    def annotate_openpose(
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
