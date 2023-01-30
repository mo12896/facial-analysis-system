from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from datahandler.dataloader import visual_dataloader
from datahandler.dataprocessor import face_detector, face_tracker
from datahandler.datawriter import video_datawriter
from datahandler.visualizer import Visualizer
from tqdm import tqdm
from utils.app_enums import VideoCodecs
from utils.constants import DATA_DIR
from utils.logger import setup_logger, with_logging

logger = setup_logger("ctrl_logger", file_logger=True)


@with_logging(logger)
def controller(args):
    """Main controller for the application."""

    configs_path = args.config

    try:
        configs: dict[str, Any] = yaml.safe_load(configs_path.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    # Construct necessary objects
    frame_loader = visual_dataloader.VideoDataLoader(
        Path(configs["VIDEO_PATH"]) / "clip_0_10570_12012.mp4"
    )
    face_detect = face_detector.create_face_detector(detector="retinaface")
    face_track = face_tracker.DlibTracker(
        face_detector=face_detect, detection_frequency=configs["DETECT_FREQ"]
    )
    video_writer = video_datawriter.VideoDataWriter(
        output_path=DATA_DIR,
        fps=frame_loader.fps,
        video_codec=VideoCodecs[configs["VIDEO_CODEC"]],
    )
    # pose_est = pose_estimator.create_pose_estimator(estimator="light_openpose")

    frame_count: int = 0

    for _, frame in enumerate(
        tqdm(frame_loader, desc="Loading frames", total=frame_loader.total_frames)
    ):
        # TODO: Resizing the image!?
        frame_cpy = np.copy(frame)

        if not frame_cpy.any():
            break

        img_info = {}
        height, width = frame.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        _, bboxes = face_track.track_faces(frame_cpy, frame_count, img_info)
        frame_count += 1
        # for crop in face_crops:
        #    emotions = emotion_detect.detect_emotions(crop)

        visualizer = Visualizer(frame)
        visualizer.draw_bboxes(bboxes)

        video_writer.write_video(visualizer.image)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("Finished processing")
    # Release the video capture object and close all windows
    frame_loader.cap.release()
    video_writer.frame_writer.release()
    cv2.destroyAllWindows()
