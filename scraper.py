import os
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Dict

import dropbox
import pandas as pd
import yaml
from dotenv import load_dotenv
from dropbox import Dropbox
from dropbox.exceptions import AuthError

from src.emotion.app_controller import Runner
from src.emotion.embedder import FaceClusterer
from src.emotion.utils.constants import (
    CONFIG_DIR,
    DATA_DIR,
    DATA_DIR_IMAGES,
    IDENTITY_DIR,
)

dropbox_folder = "/Cleaned_Team Data"

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
    "team_12",
    # "team_13",
    # "team_14",
    # "team_15",
    # "team_16",
    # "team_17",
    # "team_18",
    # "team_19",
    # "team_20",
    # "team_21",
    # "team_22",
]

days = [
    # "2023-01-10",
    "2023-01-12",
    # "2023-01-13",
]


def dropbox_connect() -> Dropbox:
    """Create a connection to Dropbox."""
    load_dotenv()
    DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print("Error connecting to Dropbox with access token: " + str(e))
    return dbx


def dropbox_list_files(dbx: Dropbox, path: str) -> pd.DataFrame:
    """Return a Pandas dataframe of files in a given Dropbox folder path in the Apps directory."""

    try:
        files = dbx.files_list_folder(path).entries
        files_list = []
        for file in files:
            if isinstance(file, dropbox.files.FileMetadata):
                metadata = {
                    "name": file.name,
                    "path_display": file.path_display,
                    "client_modified": file.client_modified,
                    "server_modified": file.server_modified,
                }
                files_list.append(metadata)

        df = pd.DataFrame.from_records(files_list)
        return df.sort_values(by="server_modified", ascending=False)

    except Exception as e:
        print("Error getting list of files from Dropbox: " + str(e))


if __name__ == "__main__":
    # set up Dropbox API client
    dbx = dropbox_connect()

    # test_folder = dropbox_folder + "/" + teams[0] + "/" + days[0]
    # files = dropbox_list_files(dbx, test_folder)

    for team in teams:
        dropbox_folder_path = dropbox_folder + "/" + team

        # download the folder and its contents to a local directory
        local_folder_path = DATA_DIR

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_zip_path = Path(temp_dir) / "temp.zip"
            dbx.files_download_zip_to_file(str(temp_zip_path), dropbox_folder_path)

            shutil.unpack_archive(str(temp_zip_path), extract_dir=local_folder_path)

        # extract the features
        for day in days:
            day_folder_path = local_folder_path / (team + "/" + day)
            output_path = IDENTITY_DIR / (team + "/" + day)
            output_path.mkdir(parents=True, exist_ok=True)

            try:
                configs: Dict[str, Any] = yaml.safe_load(
                    (CONFIG_DIR / "config.yaml").read_text()
                )
                print("Loaded config file into python dict!")
            except yaml.YAMLError as exc:
                print(exc)

            configs["EMBEDDB"] = True
            configs["ANCHOR_EMBEDDINGS"] = "embeddings_" + team + "_" + day + ".db"

            for video in day_folder_path.glob("**/*.mp4"):
                # Simulate user input for FaceClusterer
                original_stdin = sys.stdin
                user_input = "y\ny\ny\ny\n"
                sys.stdin = StringIO(user_input)

                configs["VIDEO"] = team + "/" + day + "/" + video.stem + ".mp4"

                if configs["EMBEDDB"]:
                    clusterer = FaceClusterer(configs)
                    clusterer.create_database()

                dest_folder = output_path / "images"
                if not dest_folder.is_dir():
                    src_folder = DATA_DIR_IMAGES
                    shutil.copytree(src_folder, dest_folder)

                runner = Runner(configs)
                runner.run()

                configs["EMBEDDB"] = False

                # Restore the original value of sys.stdin
                sys.stdin = original_stdin

        # delete the local folder
        shutil.rmtree(local_folder_path / team)
