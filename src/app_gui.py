import argparse


def parse_arguments() -> argparse.ArgumentParser:
    """Parse command line arguments.

    Returns:
        argparse.ArgumentParser: Parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="config.json",
        help="Path to config file (default: config.json).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser.parse_args()
