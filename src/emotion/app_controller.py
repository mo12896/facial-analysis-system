from pathlib import Path

import cv2
from datahandler.dataprocessor.face_detector import create_face_detector
from datahandler.dataprocessor.face_embedder import create_face_embedder
from datahandler.dataprocessor.face_filter import ReIdentification
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

logger = setup_logger("runner_logger", file_logger=True)


class Runner:
    def __init__(self, args):
        self.args = args

        self.detection_frequency = self.args.get("DETECTION_FREQUENCY", 10)
        self.video_path = str(DATA_DIR / self.args.get("VIDEO", "short_clip.mp4"))
        self.video_codec = self.args.get("VIDEO_CODEC", "MP4V")
        self.detector = self.args.get("DETECTOR", "retinaface")
        self.embeddings_path = str(
            DATA_DIR / self.args.get("ANCHOR_EMBDDINGS", "database/embeddings.db")
        )
        self.tracker_params = self.args.get("TRACKER", "byte")
        self.embed_params = self.args.get("EMBEDDER", "insightface")

        # Instaniate necessary objects
        self.video_info = VideoInfo.from_video_path(self.video_path)
        self.video_loader = VideoDataLoader(Path(self.video_path))
        self.face_detector = create_face_detector(self.detector)
        self.face_tracker = create_tracker(self.tracker_params)
        self.face_embedder = create_face_embedder(self.embed_params)
        self.face_reid = ReIdentification(self.embeddings_path, self.face_embedder)
        self.box_annotator = BoxAnnotator(color=Color.red())

    @with_logging(logger)
    def run(self):
        frame_count: int = 0
        curr_detections = None

        with VideoDataWriter(
            output_path=DATA_DIR,
            logger=logger,
            video_info=self.video_info,
            video_codec=VideoCodecs[self.video_codec],
        ) as video_writer:
            for _, frame in enumerate(
                tqdm(
                    self.video_loader,
                    desc="Loading frames",
                    total=self.video_loader.total_frames,
                )
            ):

                if frame_count == 0 or not frame_count % self.detection_frequency:

                    detections = self.face_detector.detect_faces(frame)

                    detections = self.face_reid.filter(detections, frame)

                    detections = self.face_tracker.track_faces(detections, frame)

                    frame_count += 1

                    frame = self.box_annotator.annotate(frame, detections)

                    video_writer.write_frame(frame)

                    curr_detections = detections

                    # if the `q` key was pressed, break from the loop
                    # key = cv2.waitKey(1) & 0xFF
                    # if key == ord("q"):
                    #     break
                else:
                    frame = self.box_annotator.annotate(frame, curr_detections)

                    video_writer.write_frame(frame)

                    frame_count += 1

        # Release the video capture object and close all windows
        self.video_loader.cap.release()
        cv2.destroyAllWindows()
