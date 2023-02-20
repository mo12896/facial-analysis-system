import logging
from pathlib import Path

import cv2
import numpy as np

from src.emotion.features.video.video_info import VideoInfo
from src.emotion.utils.app_enums import VideoCodecs


class VideoDataWriter:
    def __init__(
        self,
        output_path: Path,
        logger: logging.Logger,
        video_info: VideoInfo,
        video_codec: VideoCodecs = VideoCodecs.MP4V,
    ):
        self.output_path = output_path
        self.logger = logger
        self.fourcc = cv2.VideoWriter_fourcc(*video_codec.name)
        self.filename = "output" + video_codec.value
        self.video_info = video_info

    def __enter__(self):
        """
        Opens the output file and returns the VideoWriter instance.
        """
        self.writer = cv2.VideoWriter(
            str(self.output_path / self.filename),
            self.fourcc,
            self.video_info.fps,
            self.video_info.resolution,
        )
        self.logger.info(f"Opened output file at {self.output_path / self.filename}")
        return self

    def write_frame(self, frame: np.ndarray):
        """
        Writes a frame to the output file.
        """
        self.writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the output file.
        """
        self.writer.release()
