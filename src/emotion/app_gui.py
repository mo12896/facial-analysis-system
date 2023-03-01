import argparse
from typing import Any, Dict

import yaml

from src.emotion.utils.constants import CONFIG_DIR


def parse_arguments() -> Dict:
    """Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=CONFIG_DIR / "config.yaml",
        help="Path to config file (default: config.yaml).",
    )
    parser.add_argument(
        "-v",
        "--video",
        dest="video",
        default="test_video.mp4",
        help="Name of the video to process.",
    )
    parser.add_argument(
        "-e",
        "--embeddb",
        dest="embeddb",
        action="store_true",
        help="Set to true if you want to create a new embedding database.",
    )
    parser.add_argument(
        "-d",
        "--dashboard",
        dest="dashboard",
        action="store_true",
        help="Enable debug mode.",
    )

    args = parser.parse_args()

    try:
        configs: Dict[str, Any] = yaml.safe_load(args.config.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    configs["VIDEO"] = args.video
    configs["EMBEDDB"] = args.embeddb
    configs["DASHBOARD"] = args.dashboard

    return configs
