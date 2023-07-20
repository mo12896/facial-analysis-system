import argparse
from typing import Any, Dict

import yaml

from src.emotion.utils.constants import CONFIG_DIR


def parse_arguments() -> Dict[str, Any]:
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
        "-i",
        "--input_video",
        dest="input_video",
        help="Name of the video to process.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="",
        help="""
        Specifies which processes to run. The argument is a string that can contain any combination of the
        following digits, in increasing order:

        '0': Run the automatic template embedding generator.
        '1': Run the facial analysis pipeline.
        '2': Run the feature extractor.
        '3': Run the PERMA predictor.

        The possible combinations are:

        '0': Only runs the automatic template embedding generator.
        '1': Only runs the facial analysis pipeline.
        '2': Only runs the feature extractor.
        '3': Only runs the PERMA predictor.
        '01': Runs the automatic template embedding generator, then the facial analysis pipeline.
        '012': Runs the automatic template embedding generator, then the facial analysis pipeline, and finally the feature extractor.
        '0123': Runs all processes: template generator, facial analysis pipeline, feature extractor, and PERMA predictor.

        The processes are run in the order they are specified in the 'MODE' string.
        """,
    )
    parser.add_argument(
        "-d",
        "--dashboard",
        dest="dashboard",
        action="store_true",
        help="Include this flag to run the dashboard.",
    )
    parser.add_argument(
        "-o",
        "--output_video",
        dest="output_video",
        action="store_true",
        help="Include this flag to output a video with the results of the facial analysis pipeline. ",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Make the operation more talkative.",
    )

    args = parser.parse_args()

    # Load config file
    try:
        configs: Dict[str, Any] = yaml.safe_load(args.config.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    # Set command line arguments
    configs["MODE"] = args.mode
    configs["DASHBOARD"] = args.dashboard
    configs["OUTPUT_VIDEO"] = args.output_video

    return configs
