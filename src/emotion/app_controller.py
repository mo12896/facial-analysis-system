from pathlib import Path
from typing import Any

import cv2
import yaml
from datahandler.dataprocessor.face_detector import create_face_detector
from datahandler.dataprocessor.face_embedder import create_face_embedder
from datahandler.dataprocessor.face_filter import Filter
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

    detection_frequency = configs["DETECTION_FREQUENCY"]
    video_path = configs["VIDEO_PATH"]
    video_codec = configs["VIDEO_CODEC"]
    detector = configs["DETECTOR"]
    embeddings_path = configs["EMBEDDINGS_PATH"]

    # Instanciate necessary objects
    video_info = VideoInfo.from_video_path(video_path)
    video_loader = VideoDataLoader(Path(video_path))
    face_detector = create_face_detector(detector)
    tracker_params = {"type": "byte"}
    face_tracker = create_tracker(tracker_params)
    embed_params = {"type": "insightface", "ctx_id": 0, "det_size": (128, 128)}
    face_embedder = create_face_embedder(embed_params)
    face_filter = Filter(Path(embeddings_path), face_embedder)
    box_annotator = BoxAnnotator(color=Color.red())
    # pose_est = pose_estimator.create_pose_estimator(estimator="light_openpose")

    frame_count: int = 0
    curr_detections = None

    with VideoDataWriter(
        output_path=DATA_DIR,
        logger=logger,
        video_info=video_info,
        video_codec=VideoCodecs[video_codec],
    ) as video_writer:
        for _, frame in enumerate(
            tqdm(video_loader, desc="Loading frames", total=video_loader.total_frames)
        ):
            # TODO: Resizing the image!?
            # frame_cpy = np.copy(frame)

            # if not frame_cpy.any():
            #     break

            if frame_count == 0 or not frame_count % detection_frequency:

                detections = face_detector.detect_faces(frame)

                detections = face_filter.filter(detections, frame)

                detections = face_tracker.track_faces(detections, frame)

                frame_count += 1

                frame = box_annotator.annotate(frame, detections)

                video_writer.write_frame(frame)

                curr_detections = detections

                # if the `q` key was pressed, break from the loop
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q"):
                #     break
            else:
                frame = box_annotator.annotate(frame, curr_detections)

                video_writer.write_frame(frame)

                frame_count += 1

    # Release the video capture object and close all windows
    video_loader.cap.release()
    cv2.destroyAllWindows()
