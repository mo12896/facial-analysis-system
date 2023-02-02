import functools
import random
from pathlib import Path
from time import perf_counter

import cv2
import insightface
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
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
def crop_random_faces_from_single_frame(video_path: str, rand: bool = True) -> dict:
    """Crop random faces from a video.

    Args:
        video_path (_type_): Path to the video.
        output_folder (_type_): Path to the output folder.
        num_frames (int, optional): Number of frames to sample. Defaults to 10.
    """

    cap = cv2.VideoCapture(video_path)

    detector = RetinaFace

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i = random.sample(range(total_frames), 1)[0] if rand else 525

    # Directly jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, frame = cap.read()

    faces = detector.detect_faces(frame)

    faces_dict = {}
    for i, face in enumerate(faces):
        bbox = np.array(faces[face]["facial_area"])
        face_bbox = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        faces_dict[f"bbox_{i+1}"] = face_bbox

    cap.release()

    return faces_dict


@timer
def get_face_embeddings(images: dict) -> pd.DataFrame:
    """Returns the mean embeddings of an identity.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(128, 128))

    data = {}
    for key, img in images.items():
        face = model.get(img)[0]
        embedding = np.array(face.normed_embedding, dtype=np.float32)
        data[key] = embedding

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


# TODO: For lower dimensional embeddings use other similarity measures!
@timer
def match_embeddings(df1: pd.DataFrame, df2: pd.DataFrame) -> list[tuple]:
    if df1.shape[1] != df2.shape[1]:
        raise ValueError("Embeddings must have the same dimensionality.")

    # Compute the cosine similarity between all element pairs
    distance_matrix = cosine_similarity(df1, df2)
    print(distance_matrix)

    # Use the hungarian algorithm for bipartite matching
    _, col_ind = hungarian_algorithm(distance_matrix * -1)
    # print([(row, col) for row, col in zip(row_ind, col_ind)])

    # Retrieve the corresponding labels from df1 and df2
    df1_label = df1.index.tolist()
    df2_label = df2.iloc[col_ind].index.tolist()

    return list(zip(df1_label, df2_label))


def hungarian_algorithm(cost_matrix):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def plot_n_faces(faces: dict):
    n = len(faces)
    _, axs = plt.subplots(1, n, figsize=(n * 3, 3))
    for i, (_, face) in enumerate(faces.items()):
        axs[i].imshow(face)
        axs[i].set_title(f"bbox_{i+1}")
        axs[i].axis("off")
    plt.show()


if __name__ == "__main__":
    database = DATA_DIR_DATABASE / "embeddings.db"
    video_path = "/home/moritz/Workspace/masterthesis/data/short_clip.mp4"

    faces = crop_random_faces_from_single_frame(video_path)

    df1 = read_embeddings_from_database(database)
    df2 = get_face_embeddings(faces)

    matches = match_embeddings(df1, df2)
    print(matches)

    plot_n_faces(faces)
