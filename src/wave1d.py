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
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
from scipy.signal import argrelmax
import utils.time_series as time_series
import scipy.sparse as sp
import matplotlib.pyplot as plt

from utils.plotter import Plotter
from utils.simulate import simulate_real, forward
from tqdm import tqdm

from utils.settings import Settings
from filtering.kalman import kalman_filter
from filtering.ensemble_kalman import ensemble_kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question1() -> None:
    # for plots
    settings = Settings(damping_scale=0)
    complete_series, _ = simulate_real(settings, plot=True, prefix="state_no_damping_t")

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

    # Organize as [hs; us]
    rmses = rmses[::2] + rmses[1::2]
    biases = biases[::2] + biases[1::2]
    out_question3 = np.array([biases, rmses])
    np.savetxt("table_question3.csv", out_question3, delimiter=",", fmt="%1.4f")


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
            kwargs |= {"real": observed_data[i, 1:]}
        Plotter.plot_bands(
            x, stats[0, :], stats[1, :], f"ensembles_stats_{i}.pdf", **kwargs, show=True
        )

    spreads = list(map(lambda x: x[1, :], stats_stations))
    spreads_h = spreads[:5]
    spreads_u = spreads[5:]

    Plotter.plot_spreads(x, spreads_h, spreads_u, "test.pdf", settings.names, show=True)  # type: ignore


def question5() -> None:
    settings = Settings(add_noise=True)
    settings.initialize()
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
    R = 0.1 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0 * np.ones((n_state, 1))
    initial_covariance = 1 * np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    states, covariances = kalman_filter(
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

    # Simulations without noise ("truth"?)
    settings_real = Settings(add_noise=False)
    settings_real.initialize()
    states_real, _ = simulate_real(settings_real)

    times = range(len(settings.ts))
    times = [5, 50]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t, states, covariances, observed_data, settings, real=states_real, show=True
        )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            observed_data,
            settings,
            variable_name,
            real=states_real,
            show=True,
        )


def question6() -> None:
    settings = Settings(add_noise=True)
    settings.initialize()
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
    R = 0.1 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0 * np.ones((n_state, 1))
    initial_covariance = np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    ensemble_size = 50
    states, covariances = ensemble_kalman_filter(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left,
        observed_data,
        ensemble_size,
    )

    # Simulations without noise ("truth"?)
    settings_real = Settings(add_noise=False)
    settings_real.initialize()
    states_real, _ = simulate_real(settings_real)

    # times = range(len(settings.ts))
    times = [5, 50]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            observed_data,
            settings,
            real=states_real,
            is_ensemble=True,
            show=True,
        )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            observed_data,
            settings,
            variable_name,
            real=states_real,
            is_ensemble=True,
            show=True,
        )


def question7() -> None:
    """"""

    settings = Settings(add_noise=False)
    settings.initialize()
    n_stations = len(settings.names)
    real_states, real_observations = simulate_real(settings)
    real_observations = real_observations[:5, :]

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
    R = 2 * sp.eye(n_stations, format="csr")
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0 * np.ones((n_state, 1))
    initial_covariance = 0.01 * np.eye(n_state)

    ensemble_size = 50
    states, covariances = ensemble_kalman_filter(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left,
        real_observations,
        ensemble_size,
    )

    times = range(len(settings.ts))
    # times = [5, 50]
    times = []
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            real_observations,
            settings,
            real=real_states,
            is_ensemble=True,
            show=True,
        )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            real_observations,
            settings,
            variable_name,
            real=real_states,
            is_ensemble=True,
            show=True,
        )


def question8() -> None:
    """"""

    settings = Settings(add_noise=True)
    settings.initialize()
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
    R = 5 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 2 * np.ones((n_state, 1))
    initial_covariance = np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    states, covariances = kalman_filter(
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

    # Simulations without noise ("truth"?)
    settings_real = Settings(add_noise=False)
    settings_real.initialize()
    states_real, _ = simulate_real(settings_real)

    # times = range(len(settings.ts))
    times = [5, 50]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            observed_data,
            settings,
            real=states_real,
            show=True,
        )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            observed_data,
            settings,
            variable_name,
            real=states_real,
            show=True,
        )


def question9() -> None:
    """"""

    settings = Settings(add_noise=True)
    settings.initialize()

    # Remove one (Cadzand) station
    n_stations = len(settings.names) - 1

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
    H[aux, settings.ilocs_waterlevel[1:]] = 1
    R = 5 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0 * np.ones((n_state, 1))
    initial_covariance = np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names))
    datasets = list(map(data_file, station_names))
    _, observed_data = time_series.read_datasets(datasets)

    states, covariances = kalman_filter(
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

    # times = range(len(settings.ts))
    times = [5, 50]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            observed_data,
            settings,
            show=True,
            stations_shift=1,
        )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            observed_data,
            settings,
            variable_name,
            show=True,
            shift=1,
        )


def question10() -> None:
    """"""

    settings = Settings(add_noise=True)
    settings.initialize()

    # Remove one (Cadzand) station
    n_stations = len(settings.names) - 1

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
    H[aux, settings.ilocs_waterlevel[1:]] = 1
    R = 1 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 0 * np.ones((n_state, 1))
    initial_covariance = np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names[1:]))
    datasets = list(map(storm_file, station_names))
    obs_times, observed_data = time_series.read_datasets(datasets)

    # Indexing storm peak and forecasting
    index_peak_storm = int(np.argmax(observed_data[0, :]))
    cut_index = index_peak_storm - 10

    states, covariances = kalman_filter(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left[:cut_index],
        observed_data[:, :cut_index],
    )

    # Forecast
    n_x = M.shape[0]
    forecast_times = obs_times[cut_index:]
    state_forecast, _ = forward(
        _M,
        _B,
        _H,
        _Q,
        _R,
        states[:, -1].reshape((n_x, 1)),
        settings.h_left[cut_index:],
        len(forecast_times) - 1,
        deterministic=True,
    )

    # times = range(len(settings.ts))
    # times = [5, 50]
    # for t in tqdm(times):
    #     Plotter.plot_KF_states(
    #         t,
    #         states,
    #         covariances,
    #         observed_data,
    #         settings,
    #         show=True,
    #         stations_shift=1,
    #         forecast=(index_storm_peak, state_forecast),
    #     )

    indices = [0, 40, 50, 51]
    # indices = np.arange(states.shape[0] - 1).tolist()
    for i in tqdm(indices):
        if i % 2 == 0:
            aux = str(int(i / 2))
            variable_name = "$h_{{" + aux + "}} (\mathrm{m})$"
        else:
            aux = str(int((i - 1) / 2))
            variable_name = "$u_{{" + aux + "}}\ (\mathrm{m/s})$"
        Plotter.plot_KF_time(
            i,
            states,
            covariances,
            observed_data,
            settings,
            variable_name,
            show=True,
            shift=1,
            forecast=(cut_index, state_forecast),
        )


# main program
if __name__ == "__main__":
    # question1()
    # question3()
    question4()
    # question5()
    # question6()
    # question7()
    # question8()
    # question9()
    # question10()
