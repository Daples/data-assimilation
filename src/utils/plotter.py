import os
from typing import Any, cast, Iterable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from utils._typing import DataArray
import utils.time_series as time_series
from utils.data_handler import DataHandler
from utils.settings import Settings
from utils import index_arrays


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
    args: list[Any] = ["k-o"]
    kwargs: dict[str, Any] = {"markersize": 3}
    figsize_standard: tuple[int, int] = (8, 5)
    figsize_horizontal: tuple[int, int] = (16, 5)
    figsize_vertical: tuple[int, int] = (8, 10)
    font_size: int = 18
    bands_alpha: float = 0.2
    h_label: str = "$h\ (\mathrm{m})$"
    u_label: str = "$u\ (\mathrm{m})$"
    x_label: str = "$x\ (\mathrm{km})$"
    c_label: str = "$c\ (\mathrm{m/s}))$"
    t_label: str = "Time"

    @staticmethod
    def __clear__() -> None:
        """It clears the graphic objects."""

        plt.close("all")
        plt.cla()
        plt.clf()

    @classmethod
    def __setup_config__(cls) -> None:
        """It sets up the matplotlib configuration."""

        plt.rc("text", usetex=True)
        plt.rcParams.update({"font.size": cls.font_size})

    @classmethod
    def legend(cls, ax: Axes) -> None:
        """It moved the legend outside the plot.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower right",
            ncol=3,
        )

        # Bottom outside (may overlap)
        # # Shrink current axis's height by 10% on the bottom
        # box = ax.get_position()
        # ax.set_position(
        #     [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        # )

        # # Put a legend below current axis
        # ax.legend(
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.05),
        #     ncol=5,
        # )

        # Right outside
        # # Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # # Put a legend to the right of the current axis
        # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    @classmethod
    def date_axis(cls, ax: Axes) -> None:
        """It formats the x-axis for dates.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=cls.font_size)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

    @classmethod
    def grid(cls, ax: Axes) -> None:
        """It adds a grid to the axes.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes.
        """

        ax.grid(alpha=0.4)

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
    def plot_KF_time(
        cls,
        i: int,
        estimations: DataArray,
        covariances: list[np.ndarray],
        observations: DataArray,
        settings: Settings,
        variable_label: str,
        real: DataArray | None = None,
        is_ensemble: bool = False,
        show: bool = False,
        shift: int = 0,
        forecast: tuple[int, np.ndarray] | None = None,
        prefix: str = "",
    ) -> None:
        """It plots the estimated state vector after Kalman filtering at time t.

        Parameters
        ----------
        i: int
            The location index.
        estimations: utils._typing.DataArray
            The estimated states.
        covariances: list[np.ndarray]
            The sequence of covariance matrices.
        observations: utils._typing.DataArray
            The real measurement data.
        settings: utils.settings.Settings
            The simulation settings object.
        variable_label: str
            The name of the state to plot.
        real: utils._typing.DataArray | None, optional
            States to compare to. Default: None
        show: bool, optional
            Whether to show the figures or not. Default: False
        is_ensemble: bool, optional
            If it corresponds to an ensemble Kalman filter. Default: False
        shift: int, optional
            It shifts the stations if needed (exlude the first `shift` stations).
            Default: 0
        forecast: tuple[int, numpy.ndarray] | None, optional
            The tuple of the cut index and the forecasted state. Default = None
        prefix: str, optional
            A prefix for filenames. Default: ""
        """

        cls.__clear__()
        cls.__setup_config__()
        estimations, observations = DataHandler.__cast_array__(
            estimations, observations
        )

        cut_index = len(settings.times) + 1
        forecast_state = np.zeros(0)
        if forecast is not None:
            cut_index, forecast_state = forecast

        real_state = np.zeros(0)
        if real is not None:
            real_state = DataHandler.__cast_array__(real)[0]

        measurement_point = False
        output_index = 0
        if i in settings.ilocs_waterlevel[shift:]:
            measurement_point = True
            output_index = np.where(settings.ilocs_waterlevel[shift:] == i)[0][0]

        _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)
        stds = list(map(lambda cov: np.sqrt(cov[i, i]), covariances))

        x = settings.times
        s = stds
        y = estimations[i, :]

        x, obs = index_arrays(  # type: ignore
            x,
            observations[output_index, :-1],
        )

        x_estimation = x
        if forecast is not None:
            x_forecast = x[cut_index - 2 :]
            y_forecast = forecast_state[i, :]
            ax.plot(x_forecast, y_forecast, "k", label="Forecast")
            x_estimation = x[: cut_index - 1]
            ax.axvline(x=x_forecast[0], color="k", linestyle="--", alpha=0.4)

        ax.fill_between(
            x_estimation,
            (y - s),  # type: ignore
            (y + s),  # type: ignore
            color="b",
            alpha=cls.bands_alpha,
            zorder=-1,
        )
        label = "KF"
        if is_ensemble:
            label = "EnKF"

        # Plot filter estimation
        ax.plot(x_estimation, y, "b", label=label, zorder=3)

        # Plot actual observations
        if measurement_point:
            ax.plot(
                x,
                obs,
                "o",
                markevery=1,
                markersize=2,
                color="red",
                linewidth=1,
                label="Observations",
                zorder=2,
            )

        # Plot reference data (deterministic simulation in our case)
        if real is not None:
            x, y_real = index_arrays(x, real_state[i, :])  # type: ignore
            ax.plot(x, y_real, "k", label="Deterministic", alpha=0.5, zorder=1)

        ax.set_xlabel(cls.t_label)
        ax.set_ylabel(variable_label)

        cls.grid(ax)
        cls.date_axis(ax)
        cls.legend(ax)

        name = f"{prefix}_series_{label}_at_{i}.pdf"
        if show:
            plt.show()
        else:
            plt.savefig(cls.add_folder(name), bbox_inches="tight")

    @classmethod
    def plot_KF_states(
        cls,
        t: int,
        estimations: DataArray,
        covariances: list[np.ndarray],
        observations: DataArray,
        settings: Settings,
        real: DataArray | None = None,
        show: bool = False,
        is_ensemble: bool = False,
        stations_shift: int = 0,
        prefix: str = "",
    ) -> None:
        """It plots the estimated state vector after Kalman filtering at time t.

        Parameters
        ----------
        t: int
            The time index.
        estimations: utils._typing.DataArray
            The estimated states.
        covariances: list[np.ndarray]
            The sequence of covariance matrices.
        observations: utils._typing.DataArray
            The real measurement data.
        settings: utils.settings.Settings
            The simulation settings object.
        real: utils._typing.DataArray | None, optional
            States to compare to (deterministic in our case). Default: None
        show: bool, optional
            Whether to show the figures or not. Default: False
        is_ensemble: bool, optional
            If it corresponds to an ensemble Kalman filter. Default: False
        prefix: str, optional
            A prefix for filename. Default: ""
        """

        cls.__clear__()
        cls.__setup_config__()
        estimations, observations = DataHandler.__cast_array__(
            estimations, observations
        )

        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=cls.figsize_vertical)
        xs = [settings.x_h_km, settings.x_u_km]
        variables = [cls.h_label, cls.u_label]
        stds = np.sqrt(np.diag(covariances[t]))
        label = "KF"
        if is_ensemble:
            label = "EnKF"
        for i, x in enumerate(xs):
            s = stds[i:-1:2]
            y = estimations[i:-1:2, t]
            axs[i].plot(x, y, "b", label=label)
            if i == 0:
                axs[i].scatter(
                    settings.xlocs_waterlevel[stations_shift:] / 1000,
                    observations[:, t],
                    c="b",
                    marker="x",
                    label="Observations",
                )
            axs[i].fill_between(x, (y - s), (y + s), color="b", alpha=0.2)
            if real is not None:
                y = cast(np.ndarray, real)[i::2, t]
                axs[i].plot(x, y, "k", label="Deterministic", alpha=0.6)
            axs[i].set_xlabel(cls.x_label)
            axs[i].set_ylabel(variables[i])

            cls.legend(axs[i])
            cls.grid(axs[i])

        name = f"{prefix}_state_{label}_t_{t}.pdf"
        fig.tight_layout(pad=2)
        if show:
            plt.show()
        else:
            plt.savefig(cls.add_folder(name), bbox_inches="tight")

    @classmethod
    def plot(
        cls,
        x: DataArray,
        y: DataArray,
        path: str | None = None,
        xlabel: str = "$x$",
        ylabel: str = "$y$",
        clear: bool = True,
    ) -> tuple[Figure, Axes]:
        """It creates a plot with standard formatting.

        Parameters
        ----------
        x: utils._typing.DataArray
            The data on horizontal axis.
        y: utils._typing.DataArray
            The data on vertical axis.
        path: str | None, optional
            The name to save the figure with. Default: None
        xlabel: str, optional
            The label of the horizontal axis. Default: "$x$"
        ylabel: str, optional
            The label of the vertical axis. Default: "$y$"
        clear: bool
            Whether to clear the figure or not. Default: True

        Returns
        -------
        matplotlib.figure.Figure
            The figure handle.
        matplotlib.figure.Axes
            The axes handle.
        """

        if clear:
            cls.__clear__()
            cls.__setup_config__()
        x, y = DataHandler.__cast_array__(x, y)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(x, y, *cls.args, **cls.kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cls.grid(ax)
        if path is not None:
            plt.savefig(cls.add_folder(path), bbox_inches="tight")
        return fig, ax

    @classmethod
    def plot_speed(
        cls, times: list[int], vs: list[float], exact: float, settings: Settings
    ) -> None:
        """Plot the estimated wave velocities and the exact value.

        Parameters
        ----------
        times: list[int]
            The time indices.
        vs: list[float]
            The list of estimated velocities.
        exact: float
            The exact velocity.
        settings: utils.settings.Settings
            The simulation settings object.
        """

        cls.__clear__()
        cls.__setup_config__()

        ts = [settings.times[j] for j in times]
        _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)

        ax.plot(ts, vs, *cls.args, **cls.kwargs, label="Estimation")
        ax.axhline(y=exact, color="b", linestyle="--", label="Real")

        ax.set_ylim(0, 15)
        ax.set_xlabel(cls.t_label)
        ax.set_ylabel(cls.c_label)
        cls.grid(ax)
        cls.date_axis(ax)
        cls.legend(ax)

        plt.savefig(cls.add_folder("velocities.pdf"), bbox_inches="tight")

    @classmethod
    def plot_state(
        cls,
        x: DataArray,
        i: int,
        settings: Settings,
        fig: Figure | None = None,
        legend: bool = False,
        name: str = "ffig_map",
        kwargs: dict = {"color": "k"},
        clear: bool = True,
        axs: tuple[Axes, ...] | None = None,
    ) -> tuple[Axes, ...]:
        """"""

        cls.__clear__()
        cls.__setup_config__()
        x = DataHandler.__cast_array__(x)[0]

        # Plot all waterlevels and velocities at one time
        if fig is None:
            fig = plt.figure(figsize=cls.figsize_vertical)
        if clear:
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            axs = (ax1, ax2)
        else:
            ax1, ax2 = axs  # type: ignore
        xh = settings.x_h_km
        ax1.plot(xh, x[0::2], **kwargs)
        ax1.set_xlabel(cls.x_label)
        ax1.set_ylabel(cls.h_label)
        cls.grid(ax1)

        xu = settings.x_u_km
        ax2.plot(xu, x[1::2], **kwargs)
        ax2.set_xlabel(cls.x_label)
        ax2.set_ylabel(cls.u_label)
        cls.grid(ax2)

        name = f"{name}_{i}.pdf"
        if legend:
            cls.legend(ax1)
            cls.legend(ax2)

        fig.tight_layout(pad=2)
        plt.draw()
        plt.savefig(cls.add_folder(name), bbox_inches="tight")
        plt.close()
        return (ax1, ax2)

    @classmethod
    def plot_indices(
        cls,
        series: DataArray,
        times: list[int],
        argmaxs: list[int],
        settings: Settings,
    ) -> None:
        """"""

        cls.__clear__()
        cls.__setup_config__()
        series = DataHandler.__cast_array__(series)[0]

        _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)
        for i, t in enumerate(times):
            level = series[::2, t]
            idx_last_max = argmaxs[i]

            kwargs = {}
            if i == 0:
                kwargs |= {"label": "Peaks"}
            ax.scatter(
                settings.x_h_km[idx_last_max],
                level[idx_last_max],
                s=15,
                c="k",
                zorder=3,
                **kwargs,
            )

            ax.plot(
                settings.x_h_km,
                level,
                zorder=1,
                # label=f"$t={t}\Delta t\ (\mathrm{{s}})$",
            )
        ax.set_xlabel(cls.x_label)
        ax.set_ylabel(cls.h_label)
        cls.grid(ax)
        cls.legend(ax)

        plt.savefig(cls.add_folder("argmaxs.pdf"), bbox_inches="tight")

    @classmethod
    def plot_series(
        cls, ts: list, series_data: np.ndarray, settings: Settings, obs_data: np.ndarray
    ) -> None:
        """Plot timeseries from model and observations."""

        cls.__clear__()
        cls.__setup_config__()

        loc_names = settings.loc_names
        nseries = len(loc_names)

        for i in range(nseries):
            _, ax = plt.subplots(figsize=cls.figsize_standard)
            ax.plot(ts, series_data[i, :], "b-", label="Simulation")

            ntimes = min(len(ts), obs_data.shape[1])
            ax.plot(ts[0:ntimes], obs_data[i, 0:ntimes], "k-", label="Observation")

            ax.set_xlabel(cls.t_label)
            ax.set_ylabel(cls.h_label)
            cls.legend(ax)
            cls.grid(ax)
            cls.date_axis(ax)

            name = f"{loc_names[i]}.pdf".replace(" ", "_")
            plt.savefig(cls.add_folder(name), bbox_inches="tight")

    @classmethod
    def plot_ensemble(
        cls,
        settings_noise: list[Settings],
        ensembles: list[DataArray],
    ) -> None:
        """It plots the simulated ensembles agains the real measurements.

        Parameters
        ----------
        settings_noise: list[utils.settings.Settings]
            The list of settings for the noisy simulations.
        ensembles: list[utils._typing.DataArray]
            The list of ensemble simulations.
        """

        cls.__clear__()
        cls.__setup_config__()

        ensembles_transformed = list(DataHandler.__cast_array__(*ensembles))
        settings = settings_noise[0]

        for j, station_name in enumerate(settings.names):
            cls.__clear__()
            _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)

            obs_times, obs_values = time_series.read_series(
                f"tide_{station_name.lower()}.txt"
            )
            for i, settings in enumerate(settings_noise):
                kwargs = {}
                if i == 1:
                    kwargs |= {"label": "Ensembles"}
                series_data = ensembles_transformed[i]
                ax.plot(
                    settings.times,
                    series_data[j, :],
                    color="silver",
                    alpha=0.4,
                    **kwargs,
                )
            ax.plot(obs_times, obs_values, color="blue", label="Observations")
            ax.set_xlabel(cls.t_label)
            ax.set_ylabel(cls.h_label)
            cls.grid(ax)
            cls.legend(ax)
            cls.date_axis(ax)

            plt.savefig(
                cls.add_folder(f"ensembles_{station_name}.pdf"), bbox_inches="tight"
            )

    @classmethod
    def plot_bands(
        cls,
        x: DataArray,
        y: DataArray,
        s: DataArray,
        filename: str,
        real: DataArray | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        show: bool = False,
    ) -> None:
        """"""

        x, y, s = DataHandler.__cast_array__(x, y, s)
        if real is not None:
            real = DataHandler.__cast_array__(real)[0]

        cls.__clear__()
        cls.__setup_config__()

        _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)
        ax.plot(x, y, "b", label="Mean")
        ax.fill_between(
            x, y - s, y + s, label="Spread", color="b", alpha=cls.bands_alpha  # type: ignore
        )

        if real is not None:
            ax.plot(
                x,
                real,
                "o",
                markevery=1,
                markersize=2,
                color="red",
                linewidth=1,
                label="Observations",
                zorder=2,
            )

        if xlabel is None:
            xlabel = cls.t_label
        ax.set_xlabel(xlabel)
        if ylabel is None:
            ylabel = "Ensemble spread"
        ax.set_ylabel(ylabel)

        cls.grid(ax)
        cls.legend(ax)
        cls.date_axis(ax)

        if show:
            plt.show()
        else:
            plt.savefig(cls.add_folder(f"{filename}.pdf"), bbox_inches="tight")

    @classmethod
    def plot_spreads(
        cls,
        x: DataArray,
        spreads_h: list[DataArray],
        spreads_u: list[DataArray],
        filename: str,
        labels: list[str],
        xlabel: str | None = None,
        ylabel: str | None = None,
        show: bool = False,
    ) -> None:
        """"""

        x = DataHandler.__cast_array__(x)[0]

        cls.__clear__()
        cls.__setup_config__()

        _, axs = plt.subplots(ncols=2, nrows=1, figsize=cls.figsize_horizontal)

        for i, s_h in enumerate(spreads_h):
            s_h = DataHandler.__cast_array__(s_h)[0]
            axs[0].plot(x, s_h, label=labels[i])

            if xlabel is None:
                xlabel = cls.t_label
            if ylabel is None:
                ylabel = "Ensemble spread"

            axs[0].set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel + f" {cls.h_label}")

        for i, s_u in enumerate(spreads_u):
            s_u = DataHandler.__cast_array__(s_u)[0]
            axs[1].plot(x, s_u, label=labels[i])

            if xlabel is None:
                xlabel = cls.t_label
            if ylabel is None:
                ylabel = "Ensemble spread"

            axs[1].set_xlabel(xlabel)
            axs[1].set_ylabel(ylabel + f" {cls.u_label}")

        cls.grid(axs[0])
        cls.grid(axs[1])

        cls.legend(axs[0])
        cls.legend(axs[1])

        cls.date_axis(axs[0])
        cls.date_axis(axs[1])

        if show:
            plt.show()
        else:
            plt.savefig(cls.add_folder(f"{filename}.pdf"), bbox_inches="tight")

    @classmethod
    def plot_densities(
        cls,
        stat_matrix: np.ndarray,
        settings: Settings,
        filename: str,
        xlabel: str | None = None,
        show=False,
    ) -> None:
        """"""

        cls.__clear__()
        cls.__setup_config__

        _, ax = plt.subplots(ncols=1, nrows=1, figsize=cls.figsize_standard)

        for i in range(len(settings.names)):
            s = pd.Series(stat_matrix[i, :])
            s.plot.kde(ax=ax, label=settings.names[i])

        if xlabel is None:
            xlabel = "Statistic"
        ax.set_xlabel(xlabel)
        cls.grid(ax)
        cls.legend(ax)

        if show:
            plt.show()
        else:
            plt.savefig(cls.add_folder(f"{filename}.pdf"), bbox_inches="tight")
