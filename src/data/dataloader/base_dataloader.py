from abc import ABC
from pathlib import Path


class DataLoader(ABC):
    """
    This is the abstract base class for
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load(self):
        """Load data from defined data path"""
