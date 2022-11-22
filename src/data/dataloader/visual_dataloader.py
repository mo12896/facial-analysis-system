import cv2

from pathlib import Path
from base_dataloader import BaseDataLoader


class VisualDataLoader(BaseDataLoader):
    """
    This is a class for loading visual data from a video file frame by frame.
    """

    def __init__(self, file_path, frequency=1):
        super().__init__(file_path, frequency)
        try:
            self.cap = cv2.VideoCapture(str(self.file_path))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        except Exception as exc:
            raise exc

    def __iter__(self):
        if not self.cap.isOpened():
            raise RuntimeError("Video file is not opened.")
        return self

    def __next__(self):
        for _ in range(self.frequency):
            success, image = self.cap.read()
            if not success:
                raise StopIteration
        return image
