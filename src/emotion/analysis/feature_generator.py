from typing import Dict, List, Protocol

import numpy as np
import pandas as pd


class Generator(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data.

        Args:
            df (pd.DataFrame): DataFrame to preprocess

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        ...


class FeatureGenerator:
    def __init__(self, steps: list[Generator]):
        """Constructor for the DataPreprocessor class.

        Args:
            steps (list[PreProcessor]): List of preprocessing steps
        """
        self.steps = steps

    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline for preprocessing the data.

        Args:
            data (pd.DataFrame): DataFrame to preprocess

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        for step in self.steps:
            data = step(data)
        return data


class VelocityGenerator:
    def __init__(self, negatives: bool = False):
        self.negatives = negatives

    def _calculate_derivatives(self, group: pd.DataFrame) -> pd.Series:
        x_diff = group["x_center"].diff()
        y_diff = group["y_center"].diff()

        # Calculate the velocity magnitude
        velocity = (x_diff**2 + y_diff**2) ** 0.5

        if self.negatives:
            return velocity

        return abs(velocity)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        velocities = (
            df.groupby("ClassID")
            .apply(lambda group: self._calculate_derivatives(group))
            .fillna(0)
            .reset_index(level=0, drop=True)
            .rename("Velocity")
        )

        # Concatenate the derivatives column with the original DataFrame
        df = pd.concat([df, velocities], axis=1)

        return df


class MaxEmotionGenerator:
    def __init__(
        self,
        cols: List[str] = [
            "Angry",
            "Disgust",
            "Happy",
            "Sad",
            "Surprise",
            "Fear",
            "Neutral",
        ],
    ) -> None:
        """Constructor for the MaxEmotionGenerator class.

        Args:
            cols (List[str]): List of columns
        """
        self.cols = cols

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the max emotion.

        Args:
            data (pd.DataFrame): DataFrame to calculate the max emotion

        Returns:
            pd.DataFrame: DataFrame with the max emotion
        """
        df["Max_Emotion"] = df[self.cols].idxmax(axis=1)

        return df


class VADGenerator:
    def __init__(
        self,
        mapping: Dict[str, list] = {
            "Happy": [1, 1, 1],
            "Sad": [-1, -1, -1],
            "Angry": [-1, 1, 1],
            "Fear": [-1, 1, -1],
            "Disgust": [-1, 0, -1],
            "Surprise": [0, 1, 0],
            "Neutral": [0, 0, 0],
        },
    ) -> None:
        """Constructor for the VADGenerator class.

        Args:
            cols (List[str]): List of columns
        """
        self.mapping = mapping

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the VAD.

        Args:
            df (pd.DataFrame): DataFrame to calculate the VAD

        Returns:
            pd.DataFrame: DataFrame with the VAD
        """
        # Use vectorization to calculate the weighted sum of the emotions
        emotions = df[list(self.mapping.keys())].values
        sam_values = np.array(list(self.mapping.values()))
        weighted_sam_values = np.dot(emotions, sam_values)

        # Add new columns for the valence, arousal, and dominance values
        df["Valence"] = weighted_sam_values[:, 0]
        df["Arousal"] = weighted_sam_values[:, 1]
        df["Dominance"] = weighted_sam_values[:, 2]

        return df
