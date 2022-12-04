from pathlib import Path

import cv2
import numpy as np

from emotion.utils.app_enums import VideoCodecs


class VideoDataWriter:
    def __init__(
        self, output_path: Path, fps: int, video_codec: VideoCodecs = VideoCodecs.MP4V
    ):
        self.output_path = output_path
        self.video_codec = video_codec
        self.fps = fps
        self.first_run = True

    def write_video(self, frame: np.ndarray):
        if self.first_run:
            frame_height, frame_width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec.name)
            filename = "output" + self.video_codec.value

            self.frame_writer = cv2.VideoWriter(
                str(self.output_path / filename),
                fourcc,
                self.fps,
                (frame_width, frame_height),
            )
            self.first_run = False
        self.frame_writer.write(frame)
