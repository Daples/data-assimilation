import numpy as np
import utils.time_series as time_series

from utils.plotter import Plotter
from utils.simulate import simulate_real

from utils.settings import Settings

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question3() -> None:
    settings = Settings()
    _, simulated_observations = simulate_real(settings, plot=False, prefix="state_t")

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    Plotter.plot_series(settings.times, simulated_observations, settings, observed_data)

    # Question 3
    rmses, biases = time_series.get_statistics(
        simulated_observations, observed_data, settings
    )

    out_question3 = np.array([biases, rmses])
    np.savetxt("table_question3.csv", out_question3, delimiter=",", fmt="%1.4f")
