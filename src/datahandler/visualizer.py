import cv2
import numpy as np


class Visualizer:
    def __init__(self, image: np.ndarray):
        self.image = image

    def draw_bboxes(self, bboxes: np.ndarray) -> np.ndarray:
        for (x, y, w, h) in bboxes:
            cv2.rectangle(
                self.image,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                thickness=5,
            )
        return self.image
