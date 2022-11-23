from tqdm import tqdm
import yaml
import argparse
import numpy as np
import cv2


import setup_file as setup
from data.dataloader import (
    visual_dataloader,
    face_tracker,
    face_detector,
    emotion_detector,
)


def controller(args: argparse.ArgumentParser):
    """Main controller for the application."""

    configs_path = setup.CONFIG_DIR / args.config

    try:
        configs = yaml.safe_load(configs_path.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    frame_loader = visual_dataloader.VisualDataLoader(configs["video_path"])
    face_detector = face_detector.OpenCVFaceDetector()
    face_tracker = face_tracker.DlibTracker(face_detector)
    emotion_detector = emotion_detector.DeepFaceEmotionDetector()

    frame_count == 0

    for _, frame in enumerate(
        tqdm(frame_loader, desc="Loading frames"), total=frame_loader.total_frames
    ):
        # TODO: Resizing the image!?
        image = frame
        image_cpy = np.copy(image)

        if image_cpy is None:
            break

        face_crops, _ = face_tracker.track_faces(image_cpy, frame_count)
        frame_count += 1
        for crop in face_crops:
            emotions = emotion_detector.detect_emotions(crop)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


print("Finished processing")
# Release the video capture object and close all windows
visual_dataloader.VisualDataLoader.cap.release()
cv2.destroyAllWindows()
