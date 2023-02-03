from src.emotion.app_controller import Runner
from src.emotion.app_gui import parse_arguments


def main() -> None:

    args = parse_arguments()
    runner = Runner(args)
    runner.run()


if __name__ == "__main__":
    main()
