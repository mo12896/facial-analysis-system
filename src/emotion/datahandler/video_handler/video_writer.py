from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from utils.app_enums import VideoCodecs


@dataclass
class VideoInfo:
    """Data class containing information about a video resolution, fps, and total frame count."""

    width: int
    height: int
    fps: int
    total_frames: Optional[int] = None

    @classmethod
    def from_video_path(cls, video_path: str) -> VideoInfo:
        """
        Returns a VideoInfo data class containing information about the video resolution, fps, and total frame count.
        :param video_path: str : The path of the video file.
        :return: VideoInfo : A data class containing information about the video resolution, fps, and total frame count.
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise Exception(f"Could not open video at {video_path}")

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return VideoInfo(width, height, fps, total_frames)

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


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
        # self.writer = None

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
