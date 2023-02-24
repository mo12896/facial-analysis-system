from src.emotion.app_controller import Runner
from src.emotion.app_gui import parse_arguments
from src.emotion.create_embedding_database import FaceClusterer


def main() -> None:

    args = parse_arguments()

    if args.get("EMBEDDB"):
        clusterer = FaceClusterer(args)
        clusterer.create_database()

    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
