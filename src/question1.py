import numpy as np
from scipy.signal import argrelmax

from utils.plotter import Plotter
from utils.simulate import simulate_real

from utils.settings import Settings

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question1() -> None:
    # for plots
    settings = Settings(damping_scale=0)
    complete_series, _ = simulate_real(
        settings, plot=False, prefix="state_no_damping_t"
    )

    times = list(range(3, 14))
    series = complete_series
    argmaxs = []
    for t in times:
        init_level = series[::2, t]
        argmaxs.append(argrelmax(init_level)[0][-1])

    Plotter.plot_indices(series, times, argmaxs, settings)

    vs = []
    for i, t in enumerate(times[:-1]):
        idx_last_max = argmaxs[i]
        distance_1 = settings.x_h[idx_last_max]
        time_1 = settings.ts[t]

        time_2 = settings.ts[t + 1]
        idx_last_max = argmaxs[i + 1]
        distance_2 = settings.x_h[idx_last_max]
        v = (distance_2 - distance_1) / (time_2 - time_1)
        vs.append(v)

    exact = np.sqrt(20 * 9.81)
    Plotter.plot_speed(times[:-1], vs, exact, settings)
