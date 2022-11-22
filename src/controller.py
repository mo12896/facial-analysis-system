from tqdm import tqdm
import yaml
import argparse

import setup_file as setup
from data.dataloader import (
    visual_dataloader,
)


def controller(args: argparse.ArgumentParser):
    """Main controller for the application."""

    configs_path = setup.CONFIG_DIR / args.config

    try:
        configs = yaml.safe_load(configs_path.read_text())
        print("Loaded config file into python dict!")
    except yaml.YAMLError as exc:
        print(exc)

    frame_loader = visual_dataloader.VisualDataLoader(configs["video_path"])

    for frame_idx, frame in enumerate(
        tqdm(frame_loader, desc="Loading frames"), total=frame_loader.total_frames
    ):
        pass
