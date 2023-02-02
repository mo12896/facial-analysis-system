import functools
import sqlite3
from time import perf_counter
from typing import Callable


class SQLite:
    """Context manager for SQLite database connections."""

    def __init__(self, file="sqlite.db"):
        self.file = file

    def __enter__(self):
        self.conn = sqlite3.connect(self.file)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()

    def __exit__(self, type, value, traceback):
        self.conn.commit()
        self.conn.close()


def timer(func: Callable) -> Callable:
    """Simple Decorator for timing arbitrary functions.

    Args:
        func (Callable): Function to be timed.

    Returns:
        _type_: _description_
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Entering function {func.__name__}...")
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to execute.")
        return result

    return wrapper
