import csv

# import os
# import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)

from src.emotion.utils.constants import IDENTITY_DIR


def concatenate_csv_files(folder_path: Path):

    files = [file_path for file_path in folder_path.glob("*.csv")]
    df_concat = pd.DataFrame()

    # Loop over each file
    for i in range(len(files) - 1):
        # Read the CSV file into a dataframe
        if i == 0:
            df_concat = pd.read_csv(files[i])

        df = pd.read_csv(files[i + 1])

        # Find the last frame number in the concatenated dataframe so far
        last_frame = df_concat["Frame"].max()

        # Add 5 to last_frame to get the starting frame number for the current dataframe
        df["Frame"] = df["Frame"] + last_frame + 5

        # Concatenate the current dataframe with the concatenated dataframe
        df_concat = pd.concat([df_concat, df], ignore_index=True)

    # Write the concatenated dataframe to a CSV file
    df_concat.to_csv(
        folder_path
        / ("{}_{}".format(folder_path.parts[-2], folder_path.parts[-1]) + ".csv"),
        index=False,
    )


if __name__ == "__main__":
    # List of input CSV files

    teams = [
        # "team_01",
        # "team_02",
        # "team_03",
        # "team_04",
        # "team_05",
        # "team_06",
        # "team_07",
        # "team_08",
        # "team_09",
        # "team_10",
        # "team_11",
        # "team_12",
        # "team_13",
        # "team_15",
        # "team_16",
        # "team_17",
        # "team_18",
        # "team_19",
        # "team_20",
        # "team_22",
    ]

    days = ["2023-01-10", "2023-01-12", "2023-01-13"]

    for team in teams:
        for day in days:
            folder_path = IDENTITY_DIR / team / day
            concatenate_csv_files(folder_path)
            # Check if the folder exists
            if folder_path.exists():
                # Create a new matches.csv file in the folder
                if not folder_path.joinpath("matches.csv").exists():
                    with folder_path.joinpath("matches.csv").open(
                        mode="w", newline=""
                    ) as csvfile:
                        # Create a CSV writer object
                        writer = csv.writer(csvfile)
                        # Write the header row to the CSV file
                        writer.writerow(["E-Mail-Adresse", "ClassID", "Day"])
                        day_obj = datetime.strptime(day, "%Y-%m-%d")

                        for i in range(4):
                            writer.writerow([0, 0, day_obj.day])
