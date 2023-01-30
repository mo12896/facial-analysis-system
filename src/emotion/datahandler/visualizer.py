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
                (w, h),
                (0, 0, 255),
                thickness=2,
            )
        return self.image

    def draw_skeleton(self, poses) -> np.ndarray:
        raise NotImplementedError

    def draw_dashboard(self, dashboard) -> np.ndarray:
        raise NotImplementedError
