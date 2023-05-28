#
# 1d shallow water model
#
# solves
# dh/dt + D du/dx = 0
# du/dt + g dh/dx + f*u = 0
#
# staggered discretiztation in space and central in time
#
# o -> o -> o -> o ->   # staggering
# L u  h u  h u  h  R   # element
# 0 1  2 3  4 5  6  7   # index in state vector
#
# m=1/2, 3/2, ...
#  u[n+1,m] + 0.5 g dt/dx ( h[n+1,m+1/2] - h[n+1,m-1/2]) + 0.5 dt f u[n+1,m]
# = u[n  ,m] - 0.5 g dt/dx ( h[n  ,m+1/2] - h[n  ,m-1/2]) - 0.5 dt f u[n  ,m]
# m=1,2,3,...
#  h[n+1,m] + 0.5 D dt/dx ( u[n+1,m+1/2] - u[n+1,m-1/2])
# = h[n,m] - 0.5 D dt/dx ( u[n,m+1/2] - u[n,m-1/2])

import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import bmat, csr_array
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import utils.time_series as time_series
from typing import cast, Any
import scipy.sparse as sp

from utils import add_folder
from utils.plotter import Plotter
from utils.simulate import forward, simulate_real
from tqdm import tqdm

from utils.settings import Settings
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from filtering.kalman import kalman_filter, kalman_filter_py

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"


def plot_state(
    fig: Figure,
    x: np.ndarray,
    i: int,
    settings: Settings,
    legend: bool = False,
    name: str = "ffig_map",
    kwargs: dict = {},
    clear: bool = True,
    axs: tuple[Axes, ...] | None = None,
) -> tuple[Axes, ...]:
    # plot all waterlevels and velocities at one time
    if clear:
        fig.clear()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        axs = (ax1, ax2)
    else:
        ax1, ax2 = axs  # type: ignore
    xh = settings.x_h
    ax1.plot(xh, x[0::2], **kwargs)
    ax1.set_ylabel("h")
    xu = settings.x_u
    ax2.plot(xu, x[1::2], **kwargs)
    ax2.set_ylabel("u")

    name = f"{name}_{i}.png"
    if legend:
        ax1.legend()
        ax2.legend()
    plt.savefig(add_folder(figs_folder, name), bbox_inches="tight")
    plt.draw()
    return (ax1, ax2)


def plot_series(
    ts: list, series_data: np.ndarray, settings: Settings, obs_data: np.ndarray
) -> None:
    # plot timeseries from model and observations
    loc_names = settings.loc_names
    nseries = len(loc_names)
    for i in range(nseries):
        _, ax = plt.subplots()
        ax.plot(ts, series_data[i, :], "b-", label="Simulation")
        ax.set_title(loc_names[i])
        ax.set_xlabel("time")
        ntimes = min(len(ts), obs_data.shape[1])
        ax.plot(ts[0:ntimes], obs_data[i, 0:ntimes], "k-", label="Observation")
        plt.legend()

        name = f"{loc_names[i]}.png".replace(" ", "_")
        plt.savefig(add_folder(figs_folder, name), bbox_inches="tight")


def question3() -> None:
    # for plots
    plt.close("all")
    fig, _ = plt.subplots()  # maps: all state vars at one time

    settings = Settings()
    complete_series, series_data = simulate_real(settings)

    # indices = range(3, 13)
    # for i in indices:
    #     level = complete_series[::2, i]
    #     idx_last_max = argrelmax(level)[0][-1]
    #     plt.plot(settings.x_h, level, zorder=1)
    #     plt.scatter(
    #         [settings.x_h[idx_last_max]], [level[idx_last_max]], s=10, c="k", zorder=3
    #     )
    # plt.xlabel("$x\ (m)$")
    # plt.ylabel("$h\ (m)$")
    # plt.savefig("argmaxs.pdf")

    # plt.clf()

    # vs = []
    # series = complete_series
    # for i in indices:
    #     init_level = series[::2, i]
    #     idx_last_max = argrelmax(init_level)[0][-1]
    #     distance_1 = settings.x_h[idx_last_max]
    #     time_1 = ts[i]

    #     init_level = series[::2, i + 1]
    #     time_2 = ts[i + 1]
    #     idx_last_max = argrelmax(init_level)[0][-1]
    #     distance_2 = settings.x_h[idx_last_max]
    #     v = (distance_2 - distance_1) / (time_2 - time_1)
    #     vs.append(v)

    # plt.plot(indices, vs, "-k", label="Estimation")
    # plt.axhline(y=np.sqrt(20 * 9.81), color="b", linestyle="--", label="Real")
    # plt.ylim([0, 15])
    # plt.xlabel("$x\ (m)$")
    # plt.ylabel("$c\ (m/s)$")
    # plt.savefig("velocities.pdf", bbox_inches="tight")

    # load observations
    # (obs_times, obs_values) = time_series.read_series("tide_cadzand.txt")
    # observed_data = np.zeros((len(settings.ilocs), len(obs_times)))
    # observed_data[0, :] = obs_values[:]
    # (obs_times, obs_values) = time_series.read_series("tide_vlissingen.txt")
    # observed_data[1, :] = obs_values[:]
    # (obs_times, obs_values) = time_series.read_series("tide_terneuzen.txt")
    # observed_data[2, :] = obs_values[:]
    # (obs_times, obs_values) = time_series.read_series("tide_hansweert.txt")
    # observed_data[3, :] = obs_values[:]
    # (obs_times, obs_values) = time_series.read_series("tide_bath.txt")
    # observed_data[4, :] = obs_values[:]

    # plot_series(settings.times, series_data, settings, observed_data)

    # Question 3
    # rmses = []
    # biases = []
    # for i, _ in enumerate(settings.loc_names):
    #     observations = observed_data[i, 1:]
    #     estimations = series_data[i, :]
    #     rmse = np.sqrt(np.square(np.subtract(observations, estimations)).mean())
    #     bias = np.mean(estimations - observations)
    #     rmses.append(rmse)
    #     biases.append(bias)

    # # Organize as [hs us]
    # rmses = rmses[::2] + rmses[1::2]
    # biases = biases[::2] + biases[1::2]
    # out_question3 = np.array([biases, rmses])
    # np.savetxt("table_question3.csv", out_question3, delimiter=",", fmt="%1.4f")


def question4() -> None:
    N = 50
    ensembles = []
    list_settings = []
    seeds = np.random.random_integers(1, 1000, size=N)
    names = []
    for i, seed in enumerate(seeds):
        settings = Settings(seed=seed, add_noise=True)
        _, series_data = simulate_real(settings)

        ensembles.append(series_data)
        list_settings.append(settings)
        names = settings.names

    for j, station_name in enumerate(names):
        Plotter.__clear__()
        plt.figure()
        ax = plt.gca()

        obs_times, obs_values = time_series.read_series(
            f"tide_{station_name.lower()}.txt"
        )
        for i, settings in enumerate(list_settings):
            kwargs = {}
            if i == 1:
                kwargs |= {"label": "Ensembles"}
            series_data = ensembles[i]
            plt.plot(
                settings.times, series_data[j, :], color="silver", alpha=0.2, **kwargs
            )
        plt.plot(obs_times, obs_values, color="blue", label="Observations")
        plt.xlabel("Time (s)")
        plt.ylabel("Water level (m)")
        plt.legend()
        plt.title(f"Water level at: {station_name}")
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        plt.savefig(
            Plotter.add_folder(f"ensembles_{station_name}.pdf"), bbox_inches="tight"
        )


def question5() -> None:
    settings = Settings(add_noise=True)
    x, _ = settings.initialize()
    n_stations = len(settings.names)

    # Construct standard system notation
    tilde_A = settings.A
    tilde_B = settings.B
    A = bmat([[tilde_A, None], [None, 1]])
    aux = np.zeros((tilde_B.shape[0], 1))
    aux[0] = 1
    B_rhs = bmat([[tilde_B, aux], [None, settings.alpha]])
    n_state = A.shape[0]

    C = csr_array((n_state, 1), dtype=np.int8)
    C[0, 0] = 1
    D = csr_array((n_state, 1), dtype=np.int8)
    D[-1, 0] = 1

    inv_A = inv(A)
    M = inv_A @ B_rhs
    B = inv_A @ C
    G = inv_A @ D

    H = csr_array((n_stations, n_state), dtype=np.int8)
    aux = np.arange(n_stations)
    H[aux, settings.ilocs_waterlevel] = 1
    # R = csr_array((n_stations, n_stations))
    R = 4 * sp.eye(n_stations)
    Q = 10 * settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0.01 * np.ones((n_state, 1))
    initial_covariance = 0.1 * np.eye(n_state)
    initial_covariance[-1, -1] = 10

    # Load observations
    (obs_times, obs_values) = time_series.read_series("tide_cadzand.txt")
    observed_data = np.zeros((len(settings.ilocs_waterlevel), len(obs_times)))
    observed_data[0, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_vlissingen.txt")
    observed_data[1, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_terneuzen.txt")
    observed_data[2, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_hansweert.txt")
    observed_data[3, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_bath.txt")
    observed_data[4, :] = obs_values[:]

    states, _ = kalman_filter(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left,
        observed_data,
    )

    states_py, _ = kalman_filter_py(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left,
        observed_data,
    )

    # Compare with simulations without noise ("truth?")
    settings = Settings(add_noise=False)
    x, _ = settings.initialize()
    states_real, observations_real = simulate_real(settings)

    k = 2
    plt.plot(settings.ts, states[k, :], "r", label="State space")
    plt.plot(settings.ts, states_real[k, :], "k", label="Simulation")
    plt.show()

    t = 2
    Plotter.__clear__()
    plt.plot(settings.x_h, states[:-1:2, t], "r", label="State space")
    plt.plot(settings.x_h, states_real[::2, t], "k", label="Simulation")
    plt.show()

    # Verify noise process
    # _, axs = plt.subplots(nrows=2, ncols=2)
    # noisy_boundary = cast(np.ndarray, settings.h_left + settings.forcing_noise)
    # axs[0, 0].plot(
    #     settings.ts,
    #     settings.h_left,
    #     "r",
    #     label="Real boundary",
    # )
    # axs[0, 0].plot(
    #     settings.ts,
    #     noisy_boundary,
    #     "k",
    #     label="Noisy boundary",
    # )
    # axs[0, 0].set_xlabel("$t$")
    # axs[0, 0].set_ylabel("h_left")
    # axs[0, 0].legend()
    # axs[0, 0].set_title("Input boundaries")

    # axs[0, 1].plot(
    #     settings.ts,
    #     noisy_boundary - states[0, :],
    #     color="r",
    #     label="AR(1) KF",
    # )
    # axs[0, 1].plot(
    #     settings.ts,
    #     noisy_boundary - states_py[0, :],
    #     color="b",
    #     label="AR(1) KFPy",
    # )
    # axs[0, 1].set_xlabel("$t$")
    # axs[0, 1].set_ylabel("N(k)")
    # axs[0, 1].legend()
    # axs[0, 1].set_title("Estimated AR(1)")

    # axs[1, 0].plot(
    #     settings.ts,
    #     states[0, :],
    #     color="r",
    #     label="Estimated boundary?",
    # )
    # axs[1, 0].plot(
    #     settings.ts,
    #     noisy_boundary,
    #     color="k",
    #     label="Noisy boundary",
    # )
    # axs[1, 0].set_xlabel("$t$")
    # axs[1, 0].set_ylabel("h_left + AR(1)")
    # axs[1, 0].legend()
    # axs[1, 0].set_title("Estimated boundary (OwnKF)")

    # axs[1, 1].plot(
    #     settings.ts,
    #     states_py[0, :],
    #     color="b",
    #     label="Estimated boundary (KFPy)",
    # )
    # axs[1, 1].plot(
    #     settings.ts,
    #     noisy_boundary,
    #     color="k",
    #     label="Noisy boundary",
    # )
    # axs[1, 1].set_xlabel("$t$")
    # axs[1, 1].set_ylabel("h_left + AR(1)")
    # axs[1, 1].legend()
    # axs[1, 1].set_title("Estimated boundary (KFpy)")

    # plt.show()

    # Plot KF(py) state estimations ts times
    ts = [2, 4, 10, 20]
    Plotter.__clear__()
    for t in ts:
        fig = plt.figure()
        kwargs = {"label": "Own", "color": "r"}
        axs = plot_state(fig, states[:-1, t], t, settings, kwargs=kwargs)

        kwargs = {"label": "Py", "color": "k"}
        plot_state(
            fig,
            states_py[:-1, t],
            t,
            settings,
            kwargs=kwargs,
            clear=False,
            axs=axs,
            legend=True,
        )
        plt.show()

    # Plotter.__clear__()

    ## Plot time series estimations at locations x[ks]
    # ks = [2, 10, 40]
    # cut = 16
    # for k in ks:
    #     kwargs = {"label": "Own", "color": "r"}
    #     axs = plt.plot(states[k, :cut], **kwargs)

    #     kwargs = {"label": "Py", "color": "k"}
    #     plt.plot(states_py[k, :cut], **kwargs)
    #     plt.legend()
    #     plt.show()

    ## Plot KF estimations at observation locations
    # for j, station_name in enumerate(settings.names):
    #     iloc = settings.ilocs_waterlevel[j]
    #     Plotter.__clear__()
    #     _, ax = plt.subplots(ncols=1, nrows=1)

    #     obs_times, obs_values = time_series.read_series(
    #         f"tide_{station_name.lower()}.txt"
    #     )
    #     plt.plot(
    #         settings.times,
    #         states[int(iloc), :],
    #         color="red",
    #         alpha=1,
    #         label="Estimation",
    #     )
    #     plt.plot(obs_times, obs_values, color="k", label="Observations")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Water level (m)")
    #     plt.legend()
    #     plt.title(f"Water level at: {station_name}")
    #     plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
    #     ax.xaxis.set_major_formatter(
    #         mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    #     )
    #     plt.show()
    #     plt.savefig(Plotter.add_folder(f"KF_{station_name}.pdf"), bbox_inches="tight")


# main program
if __name__ == "__main__":
    Plotter.__setup_config__()
    # simulate()
    # question4()
    question5()
    # plt.show()
