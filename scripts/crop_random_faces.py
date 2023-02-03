"""After running this script, you have to place the croppped face into the dedicated folder to generate the identities!"""
# import os
import random

# import sys
from pathlib import Path

import cv2
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm

from src.emotion.utils.constants import DATA_DIR_IMAGES, VIDEO_PATH

# parent_folder = os.path.abspath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
# )
# sys.path.append(parent_folder)


# Number of frames to sample from the video
NUM_FRAMES = 20
# Number of identity folders
NUM_IMAGE_DIRS = 4


def cleanup(folder_path: Path) -> None:
    """Remove all files and folder in a folder.

    Args:
        output_folder (_type_): Path to the output folder.
    """
    for element in folder_path.glob("*"):
        if element.is_file():
            element.unlink()
        elif element.is_dir():
            cleanup(element)
            element.rmdir()


def crop_random_faces_from_n_frames(video_path, output_folder, num_frames=10):
    """Crop random faces from a video.

    Args:
        video_path (_type_): Path to the video.
        output_folder (_type_): Path to the output folder.
        num_frames (int, optional): Number of frames to sample. Defaults to 10.
    """

    cap = cv2.VideoCapture(video_path)

    detector = RetinaFace

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = random.sample(range(total_frames), num_frames)

    for i in tqdm(frames):
        # Directly jump to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()

        faces = detector.detect_faces(frame)
        bboxes = np.array([faces[face]["facial_area"] for face in faces])

        for j, bbox in enumerate(bboxes):
            face = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

            filename = f"face_frame_{i}_bbox_{j}.png"
            path = output_folder / filename
            cv2.imwrite(str(path), face)

    cap.release()


if __name__ == "__main__":
    video_path = VIDEO_PATH
    output_folder = DATA_DIR_IMAGES

    if output_folder.exists():
        response = input(f"{output_folder} already exists. Overwrite? [y/n] ")
        if response != "y":
            print("Script stopped by user.")
            exit()
        cleanup(output_folder)

    crop_random_faces_from_n_frames(video_path, output_folder, num_frames=NUM_FRAMES)

    for i in range(1, NUM_IMAGE_DIRS + 1):
        folder_name = f"person_id{i}"
        folder_path = output_folder / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
