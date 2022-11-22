from app_gui import parse_arguments
import controller as ctrl


def main() -> None:
    args = parse_arguments()
    ctrl.controller(args)


if __name__ == "__main__":
    main()
