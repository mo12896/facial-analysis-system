import argparse
from typing import Any

import yaml
from utils.constants import CONFIG_DIR


def parse_arguments() -> dict:
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
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable debug mode.",
    )

    args = parser.parse_args()

    try:
        configs: dict[str, Any] = yaml.safe_load(args.config.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    return configs
