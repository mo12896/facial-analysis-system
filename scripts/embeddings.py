import os
import sqlite3
import sys
from pathlib import Path

import insightface
import numpy as np

parent_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
sys.path.append(parent_folder)


from src.emotion.datahandler.dataprocessor.face_embedder import (
    FaceEmbedder,
    create_face_embedder,
)
from src.emotion.utils.constants import DATA_DIR_DATABASE, DATA_DIR_IMAGES
from src.emotion.utils.utils import SQLite

assert insightface.__version__ >= "0.3"


def dummy_embeddings():
    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    embedding3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)

    return (embedding1, embedding2, embedding3)


def generate_face_embeddings(images_path: list[Path], embedder: FaceEmbedder) -> dict:
    """Generates the embeddings for a set of identities.

    Args:
        images_path (list[Path]): List of paths to the images of a person.

    Returns:
        list[np.ndarray]: List of embeddings.
    """

    embeddings = {}
    for image_path in images_path:
        embedding = embedder.get_anchor_face_embedding(image_path)
        # Take the folder names as id for the anchor embeddings
        embeddings[str(image_path).rsplit("/", 1)[1]] = np.array(
            embedding, dtype=np.float32
        )

    return embeddings


def write_embeddings_to_database(database: Path, embeddings: dict) -> None:
    """Writes the embeddings to a database.

    Args:
        database (Path): Path to the database.
        embeddings (tuple[np.ndarray]): A tuple of embeddings.
    """
    with SQLite(str(database)) as conn:

        # Create the table to store the embeddings
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (person_id TEXT PRIMARY KEY, embedding BLOB)"
        )
        # Insert the embeddings into the database
        for person_id, embedding in embeddings.items():
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


def validate_embeddings(
    database: Path, images_path: list[Path], embedder: FaceEmbedder
):
    """Validates the embeddings.

    Args:
        embeddings (list[np.ndarray]): List of embeddings.
        database (Path): Path to the database.
        images_path (list[Path]): List of paths to the images of a person.
    """
    embeddings_from_database = read_key_embeddings_from_database(database=database)
    embeddings_from_images = generate_face_embeddings(
        images_path=images_path, embedder=embedder
    )

    assert np.allclose(
        embeddings_from_database, np.array(list(embeddings_from_images.values()))
    )


def main():
    # embeddings = dummy_embeddings()
    images_path = [item for item in DATA_DIR_IMAGES.iterdir() if item.is_dir()]
    database = DATA_DIR_DATABASE / "embeddings.db"
    # Set embedder here!
    # embedder = create_face_embedder({"type": "facerecog"})
    embedder = create_face_embedder(
        {"type": "insightface", "ctx_id": 0, "det_size": 128}
    )

    if database.exists():
        response = input(f"{database} already exists. Overwrite? [y/n] ")
        if response == "n":
            validate_embeddings(
                database=database, images_path=images_path, embedder=embedder
            )
            exit()
        database.unlink()

    embeddings = generate_face_embeddings(images_path=images_path, embedder=embedder)
    write_embeddings_to_database(database=database, embeddings=embeddings)

    validate_embeddings(database=database, images_path=images_path, embedder=embedder)

    with SQLite(str(database)) as conn:
        conn.execute("SELECT person_id, embedding FROM embeddings")
        data = {
            person_id: np.frombuffer(embedding, dtype=np.float32)
            for person_id, embedding in conn
        }
        print(data)


if __name__ == "__main__":
    main()
