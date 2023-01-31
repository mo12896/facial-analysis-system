import random
from pathlib import Path

import cv2
import insightface
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity

assert insightface.__version__ >= "0.3"


DATA_DIR_DATABASE = Path("/home/moritz/Workspace/masterthesis/data/database")


def crop_random_faces_from_single_frame(video_path) -> list:
    """Crop random faces from a video.

    Args:
        video_path (_type_): Path to the video.
        output_folder (_type_): Path to the output folder.
        num_frames (int, optional): Number of frames to sample. Defaults to 10.
    """

    cap = cv2.VideoCapture(video_path)

    detector = RetinaFace

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = random.sample(range(total_frames), 1)

    # Directly jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, frame = cap.read()

    faces = detector.detect_faces(frame)
    bboxes = np.array([faces[face]["facial_area"] for face in faces])

    faces = []
    for bbox in bboxes:
        face = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        faces.append(face)

    cap.release()

    return faces


def get_face_embeddings(images: list) -> np.ndarray:
    """Returns the mean embeddings of an identity.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(64, 64))

    images = [cv2.imread(str(image)) for image in images]

    # Predict the faces
    faces = [model.get(img)[0] for img in images]
    # Fetch the embeddings
    embeddings = [face.normed_embedding for face in faces]

    return np.array(embeddings, dtype=np.float32)


def read_embeddings_from_database():
    pass


# TODO: Make flexible for different distance metrics!
def match_embeddings(df1: pd.DataFrame, df2: pd.DataFrame):
    # Compute the cosine similarity between all element pairs
    distance_matrix = cosine_similarity(df1.iloc[:, :-1], df2.iloc[:, :-1])

    # Find the index of the minimum value in each row
    min_index = distance_matrix.argmin(axis=1)

    # Retrieve the corresponding labels from df1 and df2
    df1_label = df1.iloc[np.arange(cosine_similarity.shape[0]), -1].tolist()
    df2_label = df2.iloc[min_index, -1].tolist()

    return list(zip(df1_label, df2_label))


if __name__ == "__main__":
    database = DATA_DIR_DATABASE / "embeddings.db"
