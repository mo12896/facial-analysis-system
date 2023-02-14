from pathlib import Path

from tqdm import tqdm

from src.emotion.datahandler.dataprocessor.face_detector import create_face_detector
from src.emotion.datahandler.dataprocessor.face_embedder import create_face_embedder
from src.emotion.datahandler.dataprocessor.face_emotion_detector import (
    create_emotion_detector,
)
from src.emotion.datahandler.dataprocessor.face_filter import ReIdentification
from src.emotion.datahandler.dataprocessor.face_tracker import create_tracker
from src.emotion.datahandler.dataprocessor.head_pose_estimator import (
    create_head_pose_detector,
)
from src.emotion.datahandler.dataprocessor.pose_estimator import create_pose_estimator
from src.emotion.datahandler.video_handler.video_info import VideoInfo
from src.emotion.datahandler.video_handler.video_loader import VideoDataLoader
from src.emotion.datahandler.video_handler.video_writer import VideoDataWriter
from src.emotion.utils.annotator import BoxAnnotator
from src.emotion.utils.app_enums import VideoCodecs
from src.emotion.utils.color import Color
from src.emotion.utils.constants import DATA_DIR, IDENTITY_DIR
from src.emotion.utils.head_annotator import HeadPoseAnnotator
from src.emotion.utils.identity import IdentityHandler
from src.emotion.utils.keypoint_annotator import KeyPointAnnotator
from src.emotion.utils.logger import setup_logger, with_logging

logger = setup_logger("runner_logger", file_logger=True)


class Runner:
    def __init__(self, args):
        self.args = args

        self.detection_frequency = self.args.get("DETECTION_FREQUENCY", 5)
        self.video_path = str(DATA_DIR / self.args.get("VIDEO", "short_clip_debug.mp4"))
        self.video_codec = self.args.get("VIDEO_CODEC", "MP4V")
        self.detector = self.args.get("DETECTOR", "scrfd")
        self.embeddings_path = str(
            DATA_DIR / self.args.get("ANCHOR_EMBDDINGS", "database/embeddings.db")
        )
        self.tracker_params = self.args.get("TRACKER", "byte")
        self.emotion_detector = self.args.get("EMOTION_DETECTOR", "deepface")
        self.embed_params = self.args.get("EMBEDDER", "insightface")
        self.pose_params = self.args.get("POSE_ESTIMATOR", "l_openpose")
        self.head_pose_params = self.args.get("HEAD_POSE_ESTIMATOR", "synergy")

        # Instaniate necessary objects
        self.video_info = VideoInfo.from_video_path(self.video_path)
        self.video_loader = VideoDataLoader(Path(self.video_path))
        self.face_detector = create_face_detector(self.detector)
        self.face_tracker = create_tracker(self.tracker_params)
        self.face_embedder = create_face_embedder(self.embed_params)
        self.face_emotion_detector = create_emotion_detector(self.emotion_detector)
        self.face_reid = ReIdentification(self.embeddings_path, self.face_embedder)
        self.pose_estimator = create_pose_estimator(self.pose_params)
        self.box_annotator = BoxAnnotator(color=Color.red())
        self.keypoints_annotator = KeyPointAnnotator(color=Color.red())
        self.identities_handler = IdentityHandler()
        self.head_pose_estimator = create_head_pose_detector(self.head_pose_params)
        self.head_pose_annotator = HeadPoseAnnotator()

    @with_logging(logger)
    def run(self):

        self.on_init()

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

                    detections = self.face_emotion_detector.detect_emotions(
                        detections, frame
                    )

                    # No must have, since appearance-based tracking!
                    detections = self.face_tracker.track_faces(detections, frame)

                    detections = self.pose_estimator.estimate_poses(frame, detections)
                    detections = self.head_pose_estimator.detect_head_pose(
                        frame, detections
                    )

                    # Black background for anonymization
                    # frame[:] = 0

                    frame = self.box_annotator.annotate(frame, detections)
                    frame = self.keypoints_annotator.annotate(frame, detections)
                    frame = self.head_pose_annotator.annotate(frame, detections)

                    self.identities_handler.set_current_state(detections, frame_count)
                    self.identities_handler.write_states_to_csv()

                    video_writer.write_frame(frame)

                    curr_detections = detections

                    frame_count += 1

                else:
                    # Black background for anonymization
                    # frame[:] = 0

                    frame = self.box_annotator.annotate(frame, curr_detections)
                    frame = self.keypoints_annotator.annotate(frame, curr_detections)
                    frame = self.head_pose_annotator.annotate(frame, curr_detections)

                    video_writer.write_frame(frame)

                    frame_count += 1

        # Release the video capture object
        self.video_loader.cap.release()

    @staticmethod
    def on_init():
        """A bunch of methods which are called when the app is initialized."""
        identities = IDENTITY_DIR / "identities.csv"
        if identities.exists():
            identities.unlink()
