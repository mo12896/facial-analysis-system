import subprocess

import pandas as pd

from src.emotion.analysis.feature_vector import process, ts_feature_dict
from src.emotion.app_controller import Runner
from src.emotion.app_gui import parse_arguments
from src.emotion.embedder import FaceClusterer
from src.emotion.prediction.inference import perma_inference
from src.emotion.utils.constants import DATA_DIR_OUTPUT


def main() -> None:
    args = parse_arguments()
    video = args.get("VIDEO")

    modes = ["0", "1", "2", "3", "01", "012", "0123", "12", "23", "123"]

    if args.get("MODE") not in modes:
        raise ValueError(f"Invalid mode. Please choose from {modes}")
    if args.get("DATASET") not in ["small", "big"]:
        raise ValueError("Invalid dataset. Please choose from small, big")
    if args.get("PREDICTION") not in ["regression", "classification"]:
        raise ValueError(
            "Invalid prediction. Please choose from regression, classification"
        )

    # Template embedding generator
    if args.get("MODE") in ["0", "01", "012", "0123"]:
        verbose = args.get("VERBOSE", False)
        clusterer = FaceClusterer(args)
        clusterer.create_database(verbose=verbose)
        print("Finished with automatic template embedding generator ...")

    # Facial analysis pipeline
    if args.get("MODE") in ["1", "01", "012", "0123", "12", "123"]:
        runner = Runner(args)
        runner.run()
        print("Finished with facial analysis pipeline ...")

    # Feature extraction
    if args.get("MODE") in ["2", "012", "0123", "12", "123", "23"]:
        analysis_results = DATA_DIR_OUTPUT / (
            str(video).split(".")[0]
            + "/analysis_results/"
            + (str(video).split(".")[0] + ".csv")
        )
        extraction_results = DATA_DIR_OUTPUT / (
            str(video).split(".")[0]
            + "/extraction_results/"
            + (str(video).split(".")[0] + ".csv")
        )
        if not extraction_results.parent.exists():
            extraction_results.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(analysis_results)
        dataset = 0 if args.get("DATASET") == "small" else 1

        process(
            df,
            ts_feature_dict[dataset],
            path=extraction_results,
            save=True,
        )
        print("Finished with feature extraction ...")

    # Feature visualization
    if args.get("DASHBOARD"):
        subprocess.run(
            ["streamlit", "run", "src/emotion/dashboard.py"], capture_output=True
        )

    # PERMA prediction
    if args.get("MODE") in ["3", "23", "123", "0123"]:
        perma_inference(
            args.get("PREDICTION"), args.get("DATASET"), str(video).split(".")[0]
        )
        print("Finished with PERMA prediction ...")


if __name__ == "__main__":
    main()
