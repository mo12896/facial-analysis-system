from pathlib import Path

from tqdm import tqdm

from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.datahandler.dataprocessor.face_embedder import create_face_embedder
from src.emotion.datahandler.dataprocessor.face_filter import ReIdentification
from src.emotion.datahandler.dataprocessor.face_tracker import create_tracker
from src.emotion.datahandler.video_handler.video_info import VideoInfo
from src.emotion.datahandler.video_handler.video_loader import VideoDataLoader
from src.emotion.datahandler.video_handler.video_writer import VideoDataWriter
from src.emotion.utils.annotator import BoxAnnotator
from src.emotion.utils.app_enums import VideoCodecs
from src.emotion.utils.color import Color
from src.emotion.utils.constants import DATA_DIR
from src.emotion.utils.logger import setup_logger, with_logging

logger = setup_logger("runner_logger", file_logger=True)


class Runner:
    def __init__(self, args):
        self.args = args

        self.detection_frequency = self.args.get("DETECTION_FREQUENCY", 5)
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

                    # No must have!
                    detections = self.face_tracker.track_faces(detections, frame)

                    frame_count += 1

                    frame = self.box_annotator.annotate(frame, detections)

                    video_writer.write_frame(frame)

                    curr_detections = detections

                else:
                    frame = self.box_annotator.annotate(frame, curr_detections)

                    video_writer.write_frame(frame)

                    frame_count += 1

        # Release the video capture object and close all windows
        self.video_loader.cap.release()
