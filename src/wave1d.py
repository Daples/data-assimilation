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

from utils.plotter import Plotter
from utils.simulate import simulate_real
from tqdm import tqdm

from utils.settings import Settings
from filtering.kalman import kalman_filter
from filtering.ensemble_kalman import ensemble_kalman_filter, ensemble_kalman_filter_py

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"


def question2() -> None:
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
    _, series_data = simulate_real(settings, plot=False, prefix="state_t")

    # Load observations
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

    Plotter.plot_series(settings.times, series_data, settings, observed_data)

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
    # times = [5, 50]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t, states, covariances, observed_data, settings, real=states_real
        )

    # indices = [0, 40, 50, 51]
    indices = np.arange(states.shape[0] - 1).tolist()
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

    ensemble_size = 250
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

    ensemble_size = 500
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

    # times = range(len(settings.ts))
    times = [5, 50]
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


# main program
if __name__ == "__main__":
    # question2()
    # question3()
    # question4()
    # question5()
    # question6()
    question7()
