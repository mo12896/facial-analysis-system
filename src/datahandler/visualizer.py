import cv2
import numpy as np


class Visualizer:
    def __init__(self, image: np.ndarray, bboxes: np.ndarray):
        self.image = image
        self.bboxes = bboxes

    def draw_bboxes(self) -> np.ndarray:
        for (x, y, w, h) in self.bboxes:
            cv2.rectangle(
                self.image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )
        return self.image
