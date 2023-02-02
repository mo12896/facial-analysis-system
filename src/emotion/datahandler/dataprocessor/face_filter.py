from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize
from sklearn.metrics.pairwise import cosine_similarity
from utils.detections import Detections
from utils.utils import SQLite

from .face_embedder import FaceEmbedder


class Filter:
    """Filter class for filtering the detections."""

    def __init__(
        self,
        embeddings_path: Path,
        embedder: FaceEmbedder,
    ):
        """Constructor for the Filter class.

        Args:
            embeddings_path (Path): Path to the anchor embeddings.
            embedder (FaceEmbedder): Embedding Network for faces.
        """
        self.embeddings_path = embeddings_path
        self.embedder = embedder

    def filter(self, detections: Detections, image: np.ndarray) -> Detections:

        embeddings = self.embedder.get_face_embeddings(detections, image)

        key_embeddings = self.read_anchor_embeddings_from_database(self.embeddings_path)

        matches = self.match_embeddings(key_embeddings, embeddings)

        for match in matches:
            idx = np.where(detections.class_id == match[1])
            detections.class_id[idx] = match[0]

        return detections

    def read_anchor_embeddings_from_database(self, database: Path) -> pd.DataFrame:
        """Reads the ancor embeddings from the database.

        Args:
            database (Path): Path to the database.

        Returns:
            pd.DataFrame: Pandas DataFrame with the embeddings and their labels.
        """
        with SQLite(str(database)) as conn:
            conn.execute("SELECT person_id, embedding FROM embeddings")
            data = {
                person_id: np.frombuffer(embedding, dtype=np.float32)
                for person_id, embedding in conn
            }
            return pd.DataFrame(data).transpose()

    def match_embeddings(self, df1: pd.DataFrame, df2: pd.DataFrame) -> list[tuple]:
        """Bipartite matching of two DataFrames using the Hungarian algorithm.

        Args:
            df1 (pd.DataFrame): Pandas DataFrame 1.
            df2 (pd.DataFrame): Pandas DataFrame 2.

        Raises:
            ValueError: Raises an error if the DataFrames have different dimensions.

        Returns:
            list[tuple]: List of matches between df1 and df2.
        """
        if df1.shape[1] != df2.shape[1]:
            raise ValueError("Embeddings must have the same dimensionality.")

        # Compute the cosine similarity between all element pairs
        distance_matrix = cosine_similarity(df1, df2)

        # Use the hungarian algorithm for bipartite matching
        _, col_ind = self.hungarian_algorithm(distance_matrix * -1)

        # Retrieve the corresponding labels from df1 and df2
        df1_label = df1.index.tolist()
        df2_label = df2.iloc[col_ind].index.tolist()

        return list(zip(df1_label, df2_label))

    @staticmethod
    def hungarian_algorithm(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The Hungarian algorithm is a combinatorial optimization algorithm that solves
        the assignment problem in polynomial time.

        Args:
            cost_matrix (np.ndarray): The cost matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: The row and column indices of the optimal assignment
        """
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        return row_ind, col_ind
