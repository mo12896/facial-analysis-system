import argparse

import setup_file as setup


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default=setup.CONFIG_DIR / "config.yaml",
        help="Path to config file (default: config.yaml).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser.parse_args()
