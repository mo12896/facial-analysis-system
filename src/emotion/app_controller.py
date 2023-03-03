from pathlib import Path

from tqdm import tqdm

from src.emotion.features.annotators.annotator import BoxAnnotator
from src.emotion.features.annotators.body_annotator import BodyAnnotator
from src.emotion.features.annotators.head_annotator import HeadPoseAnnotator
from src.emotion.features.extractors.body_pose_estimator import create_pose_estimator
from src.emotion.features.extractors.face_detector import create_face_detector
from src.emotion.features.extractors.face_embedder import create_face_embedder
from src.emotion.features.extractors.face_emotion_detector import (
    create_emotion_detector,
)
from src.emotion.features.extractors.face_reid import ReIdentification

# from src.emotion.features.extractors.face_tracker import create_tracker
from src.emotion.features.extractors.gaze_detector import GazeDetector
from src.emotion.features.extractors.head_pose_estimator import (
    create_head_pose_detector,
)
from src.emotion.features.identity import IdentityHandler
from src.emotion.features.video.video_info import VideoInfo
from src.emotion.features.video.video_loader import VideoDataLoader
from src.emotion.features.video.video_writer import VideoDataWriter
from src.emotion.utils.app_enums import VideoCodecs
from src.emotion.utils.color import Color
from src.emotion.utils.constants import DATA_DIR, IDENTITY_DIR
from src.emotion.utils.logger import setup_logger, with_logging

logger = setup_logger("runner_logger", file_logger=True)


class Runner:
    def __init__(self, args):
        self.args = args

        self.detection_frequency = self.args.get("DETECTION_FREQUENCY", 5)
        self.filename = self.args.get("VIDEO", "short_clip.mp4")
        self.video_path = str(DATA_DIR / self.filename)
        self.video_codec = self.args.get("VIDEO_CODEC", "MP4V")
        self.detector = self.args.get("DETECTOR", "scrfd")
        self.embeddings_path = self.args.get("ANCHOR_EMBEDDINGS", "embeddings.db")
        # self.tracker_params = self.args.get("TRACKER", "byte")
        self.emotion_detector = self.args.get("EMOTION_DETECTOR", "deepface")
        self.embed_params = self.args.get("EMBEDDER", "insightface")
        self.pose_params = self.args.get("POSE_ESTIMATOR", "l_openpose")
        self.head_pose_params = self.args.get("HEAD_POSE_ESTIMATOR", "synergy")
        self.gaze_params = self.args.get("GAZE_DETECTOR")
        self.K = self.args.get("K", 4)

        # Instaniate necessary objects
        self.video_info = VideoInfo.from_video_path(self.video_path)
        self.video_loader = VideoDataLoader(Path(self.video_path))
        self.face_detector = create_face_detector(self.detector)
        # self.face_tracker = create_tracker(self.tracker_params)
        self.face_embedder = create_face_embedder(self.embed_params)
        self.face_emotion_detector = create_emotion_detector(self.emotion_detector)
        self.face_reid = ReIdentification(self.embeddings_path, self.face_embedder)
        self.pose_estimator = create_pose_estimator(self.pose_params)
        self.box_annotator = BoxAnnotator(color=Color.red())
        self.body_annotator = BodyAnnotator(color=Color.red())
        self.identities_handler = IdentityHandler(
            IDENTITY_DIR / (self.filename.split(".")[0] + ".csv")
        )
        self.head_pose_estimator = create_head_pose_detector(self.head_pose_params)
        self.head_pose_annotator = HeadPoseAnnotator()
        self.gaze_detector = GazeDetector(
            self.gaze_params["fov"],
            self.gaze_params["true_thresh"],
            self.gaze_params["axis_length"],
        )

    @with_logging(logger)
    def run(self):

        self.on_init()

        frame_count: int = 0
        prev_detections = None
        reid = True

        with VideoDataWriter(
            filename=str(DATA_DIR / (self.filename.split(".")[0] + "_output")),
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

                if (
                    frame_count == 0 or not frame_count % self.detection_frequency
                ) and reid:
                    # if frame_count == 0:

                    detections = self.face_detector.detect_faces(frame)

                    detections = self.face_reid.filter(detections, frame)

                    detections = self.face_emotion_detector.detect_emotions(
                        detections, frame
                    )

                    # detections = self.pose_estimator.estimate_poses(frame, detections)
                    detections = self.head_pose_estimator.detect_head_poses(
                        frame, detections
                    )

                    detections = self.gaze_detector.detect_gazes(detections)

                    # Black background for anonymization
                    # frame[:] = 0

                    frame = self.box_annotator.annotate(frame, detections)
                    # frame = self.body_annotator.annotate(frame, detections)
                    frame = self.head_pose_annotator.annotate(frame, detections)

                    self.identities_handler.set_current_state(detections, frame_count)
                    self.identities_handler.write_states_to_csv()

                    video_writer.write_frame(frame)

                    prev_detections = detections

                    frame_count += 1
                    # reid = False

                # TODO: Add Tracker
                # elif (
                #     frame_count != 0 and not frame_count % self.detection_frequency
                # ) and not reid:
                #     # Black background for anonymization
                #     # frame[:] = 0

                #     detections = self.face_detector.detect_faces(frame)
                #     # detections.tracks = prev_detections.bboxes

                #     # No must have, since ReID-based tracking!
                #     detections = self.face_tracker.track_faces(detections, frame)
                #     # detections.bboxes = detections.tracks
                #     detections.class_id = detections.tracker_id

                #     detections = self.face_emotion_detector.detect_emotions(
                #         detections, frame
                #     )

                #     detections = self.pose_estimator.estimate_poses(frame, detections)
                #     detections = self.head_pose_estimator.detect_head_poses(
                #         frame, detections
                #     )

                #     detections = self.gaze_detector.detect_gazes(detections)

                #     frame = self.box_annotator.annotate(frame, detections)
                #     frame = self.body_annotator.annotate(frame, detections)
                #     frame = self.head_pose_annotator.annotate(frame, detections)

                #     self.identities_handler.set_current_state(detections, frame_count)
                #     self.identities_handler.write_states_to_csv()

                #     video_writer.write_frame(frame)

                #     prev_detections = detections

                #     frame_count += 1
                #     if len(detections) != self.K:
                #         reid = True

                else:
                    # Black background for anonymization
                    # frame[:] = 0

                    frame = self.box_annotator.annotate(frame, prev_detections)
                    # frame = self.body_annotator.annotate(frame, prev_detections)
                    frame = self.head_pose_annotator.annotate(frame, prev_detections)

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
