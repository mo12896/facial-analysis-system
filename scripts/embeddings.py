import sqlite3
from pathlib import Path

import insightface
import mxnet as mx
import numpy as np

DATA_DIR_DATABASE = Path("/home/moritz/Workspace/masterthesis/data/database")


def dummy_embeddings():
    embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    embedding3 = np.array([0.7, 0.8, 0.9], dtype=np.float32)

    return (embedding1, embedding2, embedding3)


def get_mean_embedding(images_path: Path) -> np.ndarray:
    """Returns the mean embeddings of a person.

    Args:
        images_path (Path): Path to the images of a person.

    Returns:
        np.ndarray: Mean embedding of the person.
    """
    model = insightface.app.FaceAnalysis()

    images = [mx.image.imread(str(image)) for image in images_path.glob("*.png")]

    features = [model.get(img) for img in images]
    embedding = np.mean(features, axis=0)

    return embedding


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


def read_embeddings_from_database(database: Path) -> list[np.ndarray]:
    """Reads the embeddings from a database.

    Args:
        database (Path): Path to the database.

    Returns:
        list[np.ndarray]: List of embeddings.
    """
    with SQLite(str(database)) as conn:
        conn.execute("SELECT embedding FROM embeddings")
        embeddings = [np.frombuffer(row["embedding"], dtype=np.float32) for row in conn]

    return embeddings


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


def main():
    embeddings = dummy_embeddings()
    database = DATA_DIR_DATABASE / "embeddings.db"

    write_embeddings_to_database(database=database, embeddings=embeddings)
    embeddings = read_embeddings_from_database(database=database)

    result = np.allclose(embeddings, dummy_embeddings())
    print(True if result else False)


if __name__ == "__main__":
    main()
