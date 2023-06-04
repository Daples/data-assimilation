from typing import Any

import dateutil.parser as dtp
import numpy as np
from utils.settings import Settings
import utils.time_series as time_series

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


def read_datasets(files: list[str]) -> tuple[list[Any], np.ndarray]:
    """It reads a set of time series files.

    Parameters
    ----------
    files: list[str]
        The list of filenames.

    Returns
    -------
    list[Any]
        The time axis (assuming they are all the same).
    numpy.ndarray
        The array of observations.
    """

    (obs_times, obs_values) = time_series.read_series(files[0])
    observed_data = np.zeros((len(files), len(obs_times)))

    for i, file in enumerate(files):
        (obs_times, obs_values) = time_series.read_series(file)
        observed_data[i, :] = obs_values
    return obs_times, observed_data


def get_statistics(
    estimations: np.ndarray,
    observations: np.ndarray,
    settings: Settings,
    init_n: int = 0,
) -> tuple[list[float], ...]:
    """It calculates the statistics between estimations and real observations.

    Parameters
    ----------
    estimations: numpy.ndarray
        The estimated values.
    observations: numpy.ndarray
        The observed (reference) data. Usually the real data.
    settings: utils.settings.Settings
        The settings object.
    init_n: int, optional
        The starting index for the stations.

    Returns
    -------
    list[list[Any]]
        List of lists, where each inner list contains the value for each statistic.
    """

    rmses = []
    biases = []
    for i, _ in enumerate(settings.loc_names[init_n:]):
        observation = observations[i, 1:]
        estimation = estimations[i, :]
        rmse = np.sqrt(np.square(np.subtract(observation, estimation)).mean())
        bias = np.mean(estimation - observation)
        rmses.append(rmse)
        biases.append(bias)

    return rmses, biases


def get_statistics_ensemble(
    ensembles: list[np.ndarray], observed_data: np.ndarray, settings: Settings
) -> tuple[np.ndarray, ...]:
    """"""

    n_stations = len(settings.names)
    N = len(ensembles)
    biases_matrix = np.zeros((n_stations, N))
    rmses_matrix = np.zeros_like(biases_matrix)

    for i, ensemble in enumerate(ensembles):
        rmses, biases = time_series.get_statistics(ensemble, observed_data, settings)
        rmses_matrix[:, i] = rmses
        biases_matrix[:, i] = biases

    return rmses_matrix, biases_matrix


def get_ensemble_spread(
    ensembles_obs: list[np.ndarray],
) -> list[np.ndarray]:
    """It calculates the ensemble mean and spread at each observation station through
    time.

    Parameters
    ----------
    ensembles_obs: list[numpy.ndarray]
        The list of ensemble observations.

    Returns
    -------
    list[numpy.ndarray]
        The list of ensemble mean and std through time for each station.
    """

    n_stations, obs_time = ensembles_obs[0].shape

    stats_stations = []
    for i in range(n_stations):
        stats = np.zeros((2, obs_time))
        station_ensemble = np.array([ensemble[i, :] for ensemble in ensembles_obs])
        stats[0, :] = station_ensemble.mean(axis=0)
        stats[1, :] = station_ensemble.std(axis=0, ddof=1)
        stats_stations.append(stats)

    return stats_stations
