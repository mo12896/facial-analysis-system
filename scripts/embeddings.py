import sqlite3
from pathlib import Path

import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis

assert insightface.__version__ >= "0.3"

DATA_DIR_DATABASE = Path("/home/moritz/Workspace/masterthesis/data/database")
DATA_DIR_IMAGES = Path("/home/moritz/Workspace/masterthesis/data/images")


def dummy_embeddings():
    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    embedding3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)

    return (embedding1, embedding2, embedding3)


# TODO: Make flexible for different embedding models!
def get_mean_face_embedding(images_path: Path) -> np.ndarray:
    """Returns the mean embeddings of an identity.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(128, 128))

    images = [cv2.imread(str(image)) for image in images_path.glob("*.png")]

    # Predict the faces
    faces = [model.get(img)[0] for img in images]
    # Fetch the embeddings
    feats = [face.normed_embedding for face in faces]
    feats = np.array(feats, dtype=np.float32)
    # Average the embeddings
    embedding = np.mean(feats, axis=0)

    return embedding


def generate_face_embeddings(images_path: list[Path]) -> np.ndarray:
    """Generates the embeddings for a set of identities.

    Args:
        images_path (list[Path]): List of paths to the images of a person.

    Returns:
        list[np.ndarray]: List of embeddings.
    """

    embeddings = []
    for image_path in images_path:
        embedding = get_mean_face_embedding(image_path)
        embeddings.append(embedding)

    return np.array(embeddings, dtype=np.float32)


def write_embeddings_to_database(database: Path, embeddings: tuple[np.ndarray]):
    """Writes the embeddings to a database.

    Args:
        database (Path): Path to the database.
        embeddings (tuple[np.ndarray]): A tuple of embeddings.
    """
    with SQLite(str(database)) as conn:

        # Create the table to store the embeddings
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (person_id INTEGER PRIMARY KEY, embedding BLOB)"
        )
        # Insert the embeddings into the database
        for person_id, embedding in zip(
            list(range(1, len(embeddings) + 1)), embeddings
        ):
            query = "SELECT * FROM embeddings WHERE person_id=?"
            result = conn.execute(query, (person_id,)).fetchone()
            if result is None:
                conn.execute(
                    "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                    (person_id, sqlite3.Binary(embedding.tobytes())),
                )


def read_key_embeddings_from_database(database: Path) -> np.ndarray:
    """Reads the embeddings from a database.

    Args:
        database (Path): Path to the database.

    Returns:
        list[np.ndarray]: List of embeddings.
    """
    with SQLite(str(database)) as conn:
        conn.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row["embedding"], dtype=np.float32) for row in conn]

    return np.array(embeddings, dtype=np.float32)


class SQLite:
    """Context manager for SQLite database connections."""

    def __init__(self, file="sqlite.db"):
        self.file = file

    def __enter__(self):
        self.conn = sqlite3.connect(self.file)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()

    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()


def validate_embeddings(database: Path, images_path: list[Path]):
    """Validates the embeddings.

    Args:
        embeddings (list[np.ndarray]): List of embeddings.
        database (Path): Path to the database.
        images_path (list[Path]): List of paths to the images of a person.
    """
    embeddings_from_database = read_key_embeddings_from_database(database=database)
    embeddings_from_images = generate_face_embeddings(images_path=images_path)

    assert np.allclose(embeddings_from_database, embeddings_from_images)


def main():
    # embeddings = dummy_embeddings()
    images_path = [item for item in DATA_DIR_IMAGES.iterdir() if item.is_dir()]
    embeddings = generate_face_embeddings(images_path=images_path)
    database = DATA_DIR_DATABASE / "embeddings.db"

    if database.exists():
        response = input(f"{database} already exists. Overwrite? [y/n] ")
        if response == "n":
            validate_embeddings(database=database, images_path=images_path)
            exit()
        database.unlink()

    write_embeddings_to_database(database=database, embeddings=embeddings)

    validate_embeddings(database=database, images_path=images_path)


if __name__ == "__main__":
    main()
