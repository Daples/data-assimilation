import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
import utils.time_series as time_series
import scipy.sparse as sp

from utils.plotter import Plotter
from utils.simulate import simulate_real
from tqdm import tqdm

from utils.settings import Settings
from filtering.ensemble_kalman import ensemble_kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question8() -> None:
    settings = Settings(add_noise=False)
    settings.initialize()
    n_stations = len(settings.names)
    _, real_observations = simulate_real(settings)
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
    R = 0.1 * sp.eye(n_stations)
    Q = settings.sigma_noise**2 * G @ G.T

    # Create handles
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    # Initial state
    initial_state = 1 * np.ones((n_state, 1))
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

    times = [5, 50, 100]
    prefix = "pert_twin"
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            real_observations,
            settings,
            is_ensemble=True,
            prefix=prefix,
        )

    indices = settings.ilocs
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
            is_ensemble=True,
            prefix=prefix,
        )

    estimated_observations = states[settings.ilocs_waterlevel, :]
    rmses, biases, _ = time_series.get_statistics(
        estimated_observations, real_observations, settings
    )

    output = np.array([biases, rmses]).T
    np.savetxt("table_question8.csv", output, delimiter=",", fmt="%1.4f")
