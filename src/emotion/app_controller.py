from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from datahandler.dataprocessor.face_detector import create_face_detector
from datahandler.dataprocessor.face_tracker import create_tracker
from datahandler.video_handler.video_info import VideoInfo
from datahandler.video_handler.video_loader import VideoDataLoader
from datahandler.video_handler.video_writer import VideoDataWriter
from tqdm import tqdm
from utils.annotator import BoxAnnotator
from utils.app_enums import VideoCodecs
from utils.color import Color
from utils.constants import DATA_DIR
from utils.logger import setup_logger, with_logging

logger = setup_logger("ctrl_logger", file_logger=True)


@with_logging(logger)
def controller(args):
    """Main controller for the application."""

    configs_path = args.config

    try:
        configs: dict[str, Any] = yaml.safe_load(configs_path.read_text())
        logger.info("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        logger.info(exc)

    # Construct necessary objects
    video_info = VideoInfo.from_video_path(configs["VIDEO_PATH"])
    video_loader = VideoDataLoader(Path(configs["VIDEO_PATH"]))
    face_detector = create_face_detector(detector="retinaface")
    face_tracker = create_tracker("byte", face_detector)
    box_annotator = BoxAnnotator(color=Color.red())
    # pose_est = pose_estimator.create_pose_estimator(estimator="light_openpose")

    frame_count: int = 0

    with VideoDataWriter(
        output_path=DATA_DIR,
        logger=logger,
        video_info=video_info,
        video_codec=VideoCodecs[configs["VIDEO_CODEC"]],
    ) as video_writer:
        for _, frame in enumerate(
            tqdm(video_loader, desc="Loading frames", total=video_loader.total_frames)
        ):
            # TODO: Resizing the image!?
            frame_cpy = np.copy(frame)

            if not frame_cpy.any():
                break

            img_info = {}
            height, width = frame.shape[:2]
            img_info["height"] = height
            img_info["width"] = width

            detections = face_tracker.track_faces(frame_cpy, frame_count)
            frame_count += 1

            frame = box_annotator.annotate(frame, detections)
            video_writer.write_frame(frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    # Release the video capture object and close all windows
    video_loader.cap.release()
    cv2.destroyAllWindows()
