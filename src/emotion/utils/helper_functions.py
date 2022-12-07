from typing import Callable


def cast_list_of_strings(list_of_strings: list, destination: Callable = int) -> list:
    """Cast a list of strings to a list of ints"""
    return [
        destination(string)
        for string in list_of_strings
        if string not in ["[", "]", ",", " "]
    ]
