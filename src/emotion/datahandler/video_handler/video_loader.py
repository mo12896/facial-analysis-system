from pathlib import Path

import cv2
import numpy as np

from .base_dataloader import BaseDataLoader


class VideoDataLoader(BaseDataLoader):
    """
    This is a class for loading visual data from a video file frame by frame.
    """

    def __init__(self, video_path: Path, frequency=1):
        super().__init__(video_path, frequency)
        try:
            self.cap = cv2.VideoCapture(str(self.video_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        except Exception as exc:
            raise exc

    def __iter__(self):
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video at {self.video_path}")
        return self

    def __next__(self) -> np.ndarray:
        for _ in range(self.frequency):
            success, frame = self.cap.read()
            if not success:
                raise StopIteration
        return frame
