from typing import List, Protocol

import numpy as np
import pandas as pd


class PreProcessor(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data.

        Args:
            df (pd.DataFrame): DataFrame to preprocess

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        ...


class DataPreprocessor:
    def __init__(self, steps: list[PreProcessor]):
        """Constructor for the DataPreprocessor class.

        Args:
            steps (list[PreProcessor]): List of preprocessing steps
        """
        self.steps = steps

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Pipeline for preprocessing the data.

        Args:
            data (pd.DataFrame): DataFrame to preprocess

        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        for step in self.steps:
            data = step(data)
        return data


# Old implementation
# class LinearInterpolator:
#     def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Linearly interpolate missing frames in the data.

#         Args:
#             df (pd.DataFrame): DataFrame to interpolate

#         Returns:
#             pd.DataFrame: Interpolated DataFrame
#         """

#         # create an empty DataFrame to hold the interpolated data
#         interpolated_df = pd.DataFrame(columns=df.columns)

#         # group the DataFrame by ClassID
#         grouped = df.groupby("ClassID")

#         max_length = df["Frame"].max()

#         # iterate over each group and interpolate missing frames
#         for _, group in grouped:
#             # create a new DataFrame to hold the interpolated frames for this ClassID
#             # Iterate over the filtered data and fill in missing frames
#             last_frame = None
#             new_rows = []
#             group_length = group["Frame"].max()

#             for _, row in group.iterrows():
#                 if last_frame is None:
#                     last_frame = row
#                     new_rows.append(row)
#                 elif row["Frame"] == group_length:
#                     diff = max_length - len(new_rows)
#                     if diff:
#                         for j in range(diff):
#                             new_row = row.copy()
#                             new_row["Frame"] = row["Frame"] + j + 1
#                             new_rows.append(new_row)
#                 else:
#                     while last_frame["Frame"] < row["Frame"] - 1:
#                         missing_frame = last_frame["Frame"] + 1
#                         interp_row = last_frame.copy()
#                         interp_row["Frame"] = missing_frame
#                         for col in df.columns:
#                             if col not in ["Frame", "ClassID", "GazeDetections"]:
#                                 interp_row[col] = (
#                                     last_frame[col] * (row["Frame"] - missing_frame)
#                                     + row[col] * (missing_frame - last_frame["Frame"])
#                                 ) / (row["Frame"] - last_frame["Frame"])
#                         new_rows.append(interp_row)
#                         last_frame = interp_row
#                     new_rows.append(row)
#                     last_frame = row

#             # Combine the new rows with the original data and sort by frame
#             interpolated_df = pd.concat(
#                 [interpolated_df, pd.DataFrame(new_rows)], ignore_index=True
#             )
#             interpolated_df = interpolated_df.sort_values(by=["ClassID", "Frame"])

#         return interpolated_df


class LinearInterpolator:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Linearly interpolate missing frames in the data.

        Args:
            df (pd.DataFrame): DataFrame to interpolate

        Returns:
            pd.DataFrame: Interpolated DataFrame
        """

        # group the DataFrame by ClassID and apply the interpolation function to each group
        interpolated_groups = df.groupby("ClassID").apply(self.interpolate_group)

        # reset the index and sort by ClassID and Frame
        interpolated_df = interpolated_groups.reset_index(drop=True).sort_values(
            by=["ClassID", "Frame"]
        )

        return interpolated_df

    @staticmethod
    def interpolate_group(group):
        step = 1
        # create an array of frames to interpolate
        frames = np.arange(group["Frame"].min(), group["Frame"].max() + 1, step)
        # interpolate the missing frames using numpy.interp
        interpolated_rows = pd.DataFrame({"Frame": frames})
        for col in group.columns:
            if col not in ["Frame", "ClassID", "GazeDetections"]:
                interpolated_rows[col] = np.interp(
                    frames,
                    group["Frame"].values,
                    group[col].values,
                    left=np.nan,
                    right=np.nan,
                )
        # add the ClassID column and return the interpolated rows
        interpolated_rows["ClassID"] = group["ClassID"].iloc[0]
        return interpolated_rows


class RollingAverageSmoother:
    def __init__(self, window_size: int, cols: List[str]) -> None:
        """Constructor for the RollingAverageSmoother class.

        Args:
            window_size (int): Size of the window to use for smoothing
            cols (List[str]): List of columns to smooth
        """
        self.window_size = window_size
        self.cols = cols

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply a rolling average to the data.

        Args:
            data (pd.DataFrame): DataFrame to smooth

        Returns:
            pd.DataFrame: Smoothed DataFrame
        """

        grouped = data.groupby("ClassID")

        for _, group in grouped:

            emotions_rolling = (
                group[self.cols]
                .rolling(window=self.window_size, min_periods=1, center=True)
                .mean()
            )
            data.loc[group.index, self.cols] = emotions_rolling

        return data


class StandardDeviationNormalizer:
    def __init__(self, cols: List[str]) -> None:
        """Constructor for the Normalizer class.

        Args:
            cols (List[str]): List of columns to normalize
        """
        self.cols = cols

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data.

        Args:
            data (pd.DataFrame): DataFrame to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        grouped = data.groupby("ClassID")
        data_copy = data.copy()

        for _, group in grouped:

            norm_data = (group[self.cols] - data_copy[self.cols].mean()) / data_copy[
                self.cols
            ].std()

            data.loc[group.index, self.cols] = norm_data

        return data


class ZeroToOneNormalizer:
    def __init__(self, cols: List[str]) -> None:
        """Constructor for the Normalizer class.

        Args:
            cols (List[str]): List of columns to normalize
        """
        self.cols = cols

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data.

        Args:
            data (pd.DataFrame): DataFrame to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        grouped = data.groupby("ClassID")
        data_copy = data.copy()

        for _, group in grouped:

            norm_data = (group[self.cols] - data_copy[self.cols].min()) / (
                data_copy[self.cols].max() - data_copy[self.cols].min()
            )

            data.loc[group.index, self.cols] = norm_data

        return data


class MinusOneToOneNormalizer:
    def __init__(self, cols: List[str]) -> None:
        """Constructor for the Normalizer class.

        Args:
            cols (List[str]): List of columns to normalize
        """
        self.cols = cols

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data.

        Args:
            data (pd.DataFrame): DataFrame to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        grouped = data.groupby("ClassID")
        data_copy = data.copy()

        for _, group in grouped:

            norm_data = (group[self.cols] - data_copy[self.cols].mean()) / (
                data_copy[self.cols].max() - data_copy[self.cols].min()
            )

            data.loc[group.index, self.cols] = norm_data

        return data
