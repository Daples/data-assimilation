import os
from typing import Any, Callable

import matplotlib.pyplot as plt

from utils._typing import DataArray
from utils.data_handler import DataHandler


class Plotter:
    """A class to wrap the plotting functions.

    (Static) Attributes
    -------------------
    _folder: str
        The folder to store the output figures.
    args: list[Any]
        The additional arguments for all plots.
    kwargs: dict[str, Any]
        The keyword arguments for all plots.
    """

    _folder: str = os.path.join(os.getcwd(), "figs")
    args: list[Any] = ["-o"]
    kwargs: dict[str, Any] = {"markevery": [0, -1], "markersize": 2}

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.cla()
        plt.clf()

    @staticmethod
    def __setup_config__() -> None:
        """It sets up the matplotlib configuration."""

        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": 11})

    @classmethod
    def add_folder(cls, path: str) -> str:
        """It adds the default folder to the input path.

        Parameters
        ----------
        path: str
            A path in string.

        Returns
        -------
        str
            The path with the added folder.
        """

        if not os.path.exists(cls._folder):
            os.mkdir(cls._folder)

        return os.path.join(cls._folder, path)

    @classmethod
    def plot(
        cls,
        x: DataArray,
        y: DataArray,
        path: str,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
    ) -> None:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        path: str
            The name to save the figure with.
        xlabel: str, optional
            The label of the horizontal axis.
        ylabel: str, optional
            The label of the vertical axis.
        """

        cls.__clear__()
        cls.__setup_config__()
        x, y = DataHandler.__cast_array__(x, y)

        plt.plot(x, y, "k-o", markersize=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.savefig(cls.add_folder(path), bbox_inches="tight")
