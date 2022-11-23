from abc import ABC
from pathlib import Path


class BaseDataWriter(ABC):
    def __init__(self, file_path: Path):
        self.file_path = file_path
