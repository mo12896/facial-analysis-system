import random
from pathlib import Path

import cv2
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm


def crop_random_faces(video_path, output_folder, num_frames=10):
    """Crop random faces from a video.

    Args:
        video_path (_type_): Path to the video.
        output_folder (_type_): Path to the output folder.
        num_frames (int, optional): Number of frames to sample. Defaults to 10.
    """
    if output_folder.exists():
        response = input(f"{output_folder} already exists. Overwrite? [y/n] ")
        if response != "y":
            print("Script stopped by user.")
            exit()

    cap = cv2.VideoCapture(video_path)

    detector = RetinaFace

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = random.sample(range(total_frames), num_frames)

    for frame in tqdm(frames):
        # Directly jump to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, frame = cap.read()

        faces = detector.detect_faces(frame)
        bboxes = np.array([faces[face]["facial_area"] for face in faces])

        for j, bbox in enumerate(bboxes):
            face = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

            filename = f"face_frame_{frame}_bbox_{j}.png"
            path = output_folder / filename
            cv2.imwrite(str(path), face)

    cap.release()


if __name__ == "__main__":
    video_path = "/home/moritz/Workspace/masterthesis/data/short_clip.mp4"
    output_folder = Path("/home/moritz/Workspace/masterthesis/data/images")

    crop_random_faces(video_path, output_folder, num_frames=5)
