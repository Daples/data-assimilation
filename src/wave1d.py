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
# = h[n  ,m] - 0.5 D dt/dx ( u[n  ,m+1/2] - u[n  ,m-1/2])

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.signal import argrelmax, savgol_filter
import matplotlib.pyplot as plt
import utils.time_series as time_series

from utils import add_folder
from tqdm import tqdm

from utils.settings import Settings
from matplotlib.figure import Figure

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"


def timestep(x: np.ndarray, i: int, settings: Settings) -> np.ndarray:
    # return (h,u) one timestep later
    temp = x.copy()
    A = settings.A
    B = settings.B
    rhs = B.dot(temp)  # B*x
    rhs[0] = settings.h_left[i]  # left boundary
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

    init_times = range(1, 11)

    for t in init_times:
        plt.plot(settings.x_h, complete_series[::2, t])
    plt.plot([0.25 * settings.L] * 2, [0, 2], "--r")
    plt.savefig("test.pdf")

    smooth_complete_series = np.zeros((x.shape[0], len(ts)))
    plt.clf()

    # Smooth data
    for t in range(len(ts)):
        smooth_complete_series[:, t] = savgol_filter(
            complete_series[:, t], window_length=20, polyorder=3
        )

    for t in [1, 2, 3]:
        plt.plot(settings.x_h, smooth_complete_series[::2, t])
    plt.plot([0.25 * settings.L] * 2, [0, 2], "--r")
    plt.savefig("smooth_test.pdf")

    idxs = [1, 2, 3, 4]
    cs = []

    init_level = smooth_complete_series[:, 4]
    idx_last_max = argrelmax(init_level)[0][-1]
    distance_1 = settings.x_h[idx_last_max]
    time_1 = ts[4]

    init_level = smooth_complete_series[:, 5]
    time_2 = ts[5]
    idx_last_max = argrelmax(init_level)[0][-1]
    distance_2 = settings.x_h[idx_last_max]
    d = (distance_2 - distance_1) / (time_2 - time_1)
    time = 1
    # for j in idxs:
    #     idx = settings.ilocs[j]
    #     xloc = settings.xlocs_waterlevel[j]

    #     t_star = 0
    #     for i in np.arange(4, len(ts)):
    #         current_smooth_level = smooth_complete_series[:, i]
    #         idx_last_max = argrelmax(current_smooth_level)[0][-1]
    #         if idx_last_max > idx:
    #             t_star = ts[i]
    #             break
    #     cs.append(xloc / t_star)

    hs = smooth_complete_series[::2, :]
    us = smooth_complete_series[1::2, :]

    # cs_h = np.zeros((hs.shape[0] - 2, len(ts) - 2))
    # for i, lx in enumerate(range(1, hs.shape[0] - 1)):
    #     for j, lt in enumerate(range(1, len(ts) - 1)):
    #         d2h_dt2 = (
    #             hs[lx, lt - 1] - 2 * hs[lx, lt] + hs[lx, lt + 1]
    #         ) / settings.dt**2
    #         d2h_dx2 = (
    #             hs[lx - 1, lt] - 2 * hs[lx, lt] + hs[lx + 1, lt]
    #         ) / settings.dx**2
    #         cs_h[i, j] = np.sqrt(d2h_dt2 / d2h_dx2)

    # cs_u = np.zeros((us.shape[0] - 2, len(ts) - 2))
    # for i, lx in enumerate(range(1, us.shape[0] - 1)):
    #     for j, lt in enumerate(range(1, len(ts) - 1)):
    #         d2u_dt2 = (
    #             us[lx, lt - 1] - 2 * us[lx, lt] + us[lx, lt + 1]
    #         ) / settings.dt**2
    #         d2u_dx2 = (
    #             us[lx - 1, lt] - 2 * us[lx, lt] + us[lx + 1, lt]
    #         ) / settings.dx**2
    #         cs_u[i, j] = np.sqrt(d2u_dt2 / d2u_dx2)

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


# main program
if __name__ == "__main__":
    simulate()
    # plt.show()
