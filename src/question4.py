import numpy as np
import utils.time_series as time_series

from utils.plotter import Plotter
import matplotlib.pyplot as plt
import pandas as pd
from utils.simulate import simulate_real

from utils.settings import Settings

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question4() -> None:
    N = 50
    ensembles = []
    list_settings = []
    seeds = np.random.random_integers(1, 1000, size=N)

    for seed in seeds:
        settings = Settings(seed=seed, add_noise=True)
        _, series_data = simulate_real(settings)

        ensembles.append(series_data)
        list_settings.append(settings)

    Plotter.plot_ensemble(list_settings, ensembles)

    # Read observations
    settings = list_settings[0]
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    # Spread
    stats_stations = time_series.get_ensemble_spread(ensembles)

    x = settings.times
    for i, stats in enumerate(stats_stations):
        kwargs = {}
        if i < 5:
            kwargs |= {"real": observed_data[i, :-1]}
        Plotter.plot_bands(
            x, stats[0, :], stats[1, :], f"ensembles_stats_{i}.pdf", **kwargs
        )

    spreads = list(map(lambda x: x[1, :], stats_stations))
    spreads_h = spreads[:5]
    spreads_u = spreads[5:]

    Plotter.plot_spreads(x, spreads_h, spreads_u, "spreads", settings.names)  # type: ignore

    # Histograms
    rmses_matrix, biases_matrix = time_series.get_statistics_ensemble(
        ensembles, observed_data, settings
    )
    Plotter.plot_densities(rmses_matrix, settings, "rmses_ensemble", xlabel="RMSE")
    Plotter.plot_densities(biases_matrix, settings, "biases_ensemble", xlabel="Bias")
