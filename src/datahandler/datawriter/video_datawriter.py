from pathlib import Path

import cv2
import numpy as np


class VideoDataWriter:
    def __init__(self, output_path: Path, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.first_run = True

    def write_video(self, frame: np.ndarray):
        if self.first_run:
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (frame_width, frame_height),
            )
            self.first_run = False
        self.video_writer.write(frame)
