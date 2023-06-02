import os
from typing import Sequence
import numpy as np


def add_folder(folder: str, path: str) -> str:
    """It adds the default folder to the input path.

    Parameters
    ----------
    folder: str

    path: str
        A path in string.

    Returns
    -------
    str
        The path with the added folder.
    """

    if not os.path.exists(folder):
        os.mkdir(folder)

    return os.path.join(folder, path)


def get_index(*arrays: Sequence | np.ndarray) -> int:
    """It gets the maximum possible index for all arrays.

    Parameters
    ----------
    arrays: list[numpy.ndarray]
        The list of arrays.

    Returns
    -------
    int
        The maximum index.
    """

    lengths = [len(array) for array in arrays]
    return min(lengths)


def index_arrays(*arrays: Sequence | np.ndarray) -> tuple[Sequence | np.ndarray]:
    """It indexes arrays with the minimum length of them all.

    Parameters
    ----------
    arrays: tuple[numpy.ndarray, ...]
        The collection of arrays to index.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        The indexed arrays.
    """

    index = get_index(*arrays)
    return tuple(map(lambda x: x[:index], arrays))
