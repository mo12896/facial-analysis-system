import functools
import random
from pathlib import Path
from time import perf_counter

import cv2
import insightface
import numpy as np
import pandas as pd
from embeddings import SQLite
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
from sklearn.metrics.pairwise import cosine_similarity

assert insightface.__version__ >= "0.3"


DATA_DIR_DATABASE = Path("/home/moritz/Workspace/masterthesis/data/database")


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Entering function {func.__name__}...")
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to execute.")
        return result

    return wrapper


@timer
def crop_random_faces_from_single_frame(video_path: str) -> list:
    """Crop random faces from a video.

    Args:
        video_path (_type_): Path to the video.
        output_folder (_type_): Path to the output folder.
        num_frames (int, optional): Number of frames to sample. Defaults to 10.
    """

    cap = cv2.VideoCapture(video_path)

    detector = RetinaFace

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = random.sample(range(total_frames), 1)[0]
    # i = 400

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


@timer
def get_face_embeddings(images: list) -> pd.DataFrame:
    """Returns the mean embeddings of an identity.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(128, 128))

    # Predict the faces
    faces = [model.get(img)[0] for img in images]
    # Fetch the embeddings
    embeddings = np.array([face.normed_embedding for face in faces], dtype=np.float32)
    labels = [f"bbox_{i}" for i in range(len(embeddings))]
    data = {label: embedding for label, embedding in zip(labels, embeddings)}

    return pd.DataFrame(data).transpose()


@timer
def read_embeddings_from_database(database: Path) -> pd.DataFrame:
    with SQLite(str(database)) as conn:
        conn.execute("SELECT person_id, embedding FROM embeddings")
        data = {
            person_id: np.frombuffer(embedding, dtype=np.float32)
            for person_id, embedding in conn
        }
        return pd.DataFrame(data).transpose()


# TODO: Make flexible for different distance metrics!
@timer
def match_embeddings(df1: pd.DataFrame, df2: pd.DataFrame) -> list[tuple]:
    # Compute the cosine similarity between all element pairs
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("Embeddings must have the same dimensionality.")
    distance_matrix = cosine_similarity(df1, df2)

    # Find the index of the minimum value in each row
    min_indices = np.abs(distance_matrix).argmin(axis=1)

    # Retrieve the corresponding labels from df1 and df2
    df1_label = df1.index.tolist()
    df2_label = df2.iloc[min_indices].index.tolist()

    return list(zip(df1_label, df2_label))


if __name__ == "__main__":
    database = DATA_DIR_DATABASE / "embeddings.db"
    video_path = "/home/moritz/Workspace/masterthesis/data/short_clip.mp4"

    faces = crop_random_faces_from_single_frame(video_path)

    df1 = read_embeddings_from_database(database)
    df2 = get_face_embeddings(faces)
    matches = match_embeddings(df1, df2)

    print(matches)
