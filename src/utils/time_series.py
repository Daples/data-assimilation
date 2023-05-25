from typing import Any

import dateutil.parser as dtp

from utils import add_folder


def read_series(filename) -> tuple[list[Any], list[Any]]:
    """It reads the series file. (Implemented by Martin Verlaan)

    Parameters
    ----------
    filename: str
        The filename to read.

    Returns
    -------
    list[Any]
        The list of times.
    list[Any]
        The list of values.
    """

    infile = open(_data_folder(filename), "r")
    times = []
    values = []
    for line in infile:
        if line.startswith("#") or len(line) <= 1:
            continue
        parts = line.split()
        times.append(dtp.parse(parts[0]))
        values.append(float(parts[1]))
    infile.close()
    return (times, values)


def _data_folder(filename: str) -> str:
    """Standardize data folder location.

    Parameters
    ----------
    filename: str
        The filename to read.

    Returns
    -------
    str
        The filename with the data folder added.
    """

    return add_folder("./data/", filename)
