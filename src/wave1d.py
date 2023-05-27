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

from utils import add_folder
from utils.plotter import Plotter
from tqdm import tqdm

from utils.settings import Settings
from matplotlib.figure import Figure
from filtering.kalman import kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"


def timestep(x: np.ndarray, i: int, settings: Settings) -> np.ndarray:
    # return (h,u) one timestep later
    temp = x.copy()
    A = settings.A
    B = settings.B
    rhs = B.dot(temp)  # B*x
    rhs[0] = settings.h_left[i] + settings.forcing_noise[i]  # left boundary
    newx = spsolve(A, rhs)
    return newx


def plot_state(fig: Figure, x: np.ndarray, i: int, settings: Settings) -> None:
    # plot all waterlevels and velocities at one time
    fig.clear()
    xh = settings.x_h
    ax1 = fig.add_subplot(211)
    ax1.plot(xh, x[0::2])
    ax1.set_ylabel("h")
    xu = settings.x_u
    ax2 = fig.add_subplot(212)
    ax2.plot(xu, x[1::2])
    ax2.set_ylabel("u")

    name = f"ffig_map_{i}.png"
    plt.savefig(add_folder(figs_folder, name), bbox_inches="tight")
    plt.draw()


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


def simulate() -> None:
    # for plots
    plt.close("all")
    fig, _ = plt.subplots()  # maps: all state vars at one time

    settings = Settings()
    x, _ = settings.initialize()

    ts = settings.ts[:]  # [:40]

    series_data = np.zeros((len(settings.ilocs), len(ts)))
    complete_series = np.zeros((x.shape[0], len(ts)))
    for i in tqdm(np.arange(1, len(ts))):
        x = timestep(x, i, settings)
        # plot_state(fig, x, i, settings)
        complete_series[:, i] = x
        series_data[:, i] = x[settings.ilocs]

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
    (obs_times, obs_values) = time_series.read_series("tide_cadzand.txt")
    observed_data = np.zeros((len(settings.ilocs), len(obs_times)))
    observed_data[0, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_vlissingen.txt")
    observed_data[1, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_terneuzen.txt")
    observed_data[2, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_hansweert.txt")
    observed_data[3, :] = obs_values[:]
    (obs_times, obs_values) = time_series.read_series("tide_bath.txt")
    observed_data[4, :] = obs_values[:]

    plot_series(settings.times, series_data, settings, observed_data)

    # Question 3
    rmses = []
    biases = []
    for i, _ in enumerate(settings.loc_names):
        observations = observed_data[i, 1:]
        estimations = series_data[i, :]
        rmse = np.sqrt(np.square(np.subtract(observations, estimations)).mean())
        bias = np.mean(estimations - observations)
        rmses.append(rmse)
        biases.append(bias)

    # Organize as [hs us]
    rmses = rmses[::2] + rmses[1::2]
    biases = biases[::2] + biases[1::2]
    out_question3 = np.array([biases, rmses])
    np.savetxt("table_question3.csv", out_question3, delimiter=",", fmt="%1.4f")


def question4() -> None:
    N = 50
    ensembles = []
    list_settings = []
    seeds = np.random.random_integers(1, 1000, size=N)
    names = []
    for i, seed in enumerate(seeds):
        settings = Settings(seed=seed, add_noise=True)
        x, _ = settings.initialize()
        ts = settings.ts[:]  # [:40]

        series_data = np.zeros((len(settings.ilocs), len(ts)))
        for i in tqdm(np.arange(1, len(ts))):
            x = timestep(x, i, settings)
            series_data[:, i] = x[settings.ilocs]
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
    R = csr_array((n_stations, n_stations))
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M
    _B = lambda _: B
    _H = lambda _: H
    _Q = lambda _: Q
    _R = lambda _: R

    # Initial state
    initial_state = np.zeros((n_state, 1))
    initial_covariance = 1 * np.eye(n_state)

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

    for j, station_name in enumerate(settings.names):
        iloc = settings.ilocs_waterlevel[j]
        Plotter.__clear__()
        plt.figure()
        ax = plt.gca()

        obs_times, obs_values = time_series.read_series(
            f"tide_{station_name.lower()}.txt"
        )
        plt.plot(
            settings.times,
            states[int(iloc), :],
            color="red",
            alpha=1,
            label="Estimation",
        )
        plt.plot(obs_times, obs_values, color="k", label="Observations")
        plt.xlabel("Time (s)")
        plt.ylabel("Water level (m)")
        plt.legend()
        plt.title(f"Water level at: {station_name}")
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=10)
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        plt.savefig(Plotter.add_folder(f"KF_{station_name}.pdf"), bbox_inches="tight")


# main program
if __name__ == "__main__":
    Plotter.__setup_config__()
    # simulate()
    # question4()
    question5()
    # plt.show()
