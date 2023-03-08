import subprocess

from src.emotion.app_controller import Runner
from src.emotion.app_gui import parse_arguments
from src.emotion.embedder import FaceClusterer


def main() -> None:

    args = parse_arguments()

    if args.get("EMBEDDB"):
        clusterer = FaceClusterer(args)
        clusterer.create_database()

    runner = Runner(args)
    runner.run()

    if args.get("DASHBOARD"):
        subprocess.run(
            ["streamlit", "run", "src/emotion/dashboard.py"], capture_output=True
        )


if __name__ == "__main__":
    main()
