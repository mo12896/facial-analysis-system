from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from utils.detections import Detections


class FaceEmbedder(ABC):
    """Abstract class for face embedding extraction."""

    @abstractmethod
    def __init__(self, parameters: dict = {}):
        self.parameters = parameters

    @abstractmethod
    def get_face_embeddings(detections: Detections, image: np.ndarray) -> pd.DataFrame:
        """Retrieve the face embeddings for a set of detections.

        Args:
            detections (Detections): Set of detections stored in a Detections object.
            image (np.ndarray): The current frame of the video.

        Returns:
            pd.DataFrame: A Dataframe storing the embeddings for each detection.
        """


def create_face_embedder(parameters: dict) -> FaceEmbedder:
    """Create a face embedder based on the given parameters.

    Args:
        parameters (dict): A dictionary containing the parameters for the face embedder.

    Returns:
        FaceEmbedder: A face embedder object.
    """

    if parameters["type"] == "insightface":
        return InsightFaceEmbedder(parameters)
    else:
        raise ValueError("Unknown face embedder type.")


class InsightFaceEmbedder(FaceEmbedder):
    """Wrapper around the insightface model for face embedding extraction."""

    def __init__(self, parameters: dict = {}):
        super().__init__(parameters)
        self.model = FaceAnalysis()

        ctx_id = parameters.get("ctx_id", 0)
        det_size = parameters.get("det_size", 128)
        self.model.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def get_face_embeddings(
        self, detections: Detections, image: np.ndarray
    ) -> pd.DataFrame:
        """Get the face embeddings for a set of detections using the insightface model."""

        data = {}

        for key, bbox in zip(detections.class_id, detections.bboxes):
            img = image[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
            face = self.model.get(img)[0]
            embedding = np.array(face.normed_embedding, dtype=np.float32)
            data[key] = embedding

        return pd.DataFrame(data).transpose()
