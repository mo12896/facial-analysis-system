from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

import setup_file as setup
from datahandler.dataloader import visual_dataloader
from datahandler.dataprocessor import emotion_detector, face_detector, face_tracker
from datahandler.datawriter import video_datawriter
from datahandler.visualizer import Visualizer


def controller(args):
    """Main controller for the application."""

    configs_path = args.config

    try:
        configs: dict[str, Any] = yaml.safe_load(configs_path.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    frame_loader = visual_dataloader.VisualDataLoader(setup.DATA_DIR / "test_video.mp4")
    face_detect = face_detector.OpenCVFaceDetector(
        face_detector=cv2.CascadeClassifier(configs["FACEDETECTOR"])
    )
    face_track = face_tracker.DlibTracker(face_detect)
    emotion_detect = emotion_detector.DeepFaceEmotionDetector()
    frame_writer = video_datawriter.VideoDataWriter(
        output_path=setup.DATA_DIR / "output.mp4", fps=frame_loader.fps
    )

    frame_count: int = 0

    for _, frame in enumerate(
        tqdm(frame_loader, desc="Loading frames", total=frame_loader.total_frames)
    ):
        # TODO: Resizing the image!?
        image = frame
        image_cpy = np.copy(image)

        if image_cpy is None:
            break

        face_crops, bboxes = face_track.track_faces(image_cpy, frame_count)
        frame_count += 1
        # for crop in face_crops:
        #    emotions = emotion_detect.detect_emotions(crop)

        visualizer = Visualizer(image, bboxes)
        visualizer.draw_bboxes

        frame_writer.write_video(visualizer.image)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


print("Finished processing")
# Release the video capture object and close all windows
# visual_dataloader.VisualDataLoader.cap.release()
cv2.destroyAllWindows()
