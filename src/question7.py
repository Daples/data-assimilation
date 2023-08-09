import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
import scipy.sparse as sp
import matplotlib.pyplot as plt

from utils.plotter import Plotter
from utils.simulate import simulate_real
from tqdm import tqdm

from utils.settings import Settings
from filtering.kalman import kalman_filter
from filtering.ensemble_kalman import ensemble_kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question7() -> None:
    """"""

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
    R = 0.1 * sp.eye(n_stations, format="csr")
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

    states_kf, covariances_kf = kalman_filter(
        _M,
        _B,
        _H,
        _Q,
        _R,
        initial_state,
        initial_covariance,
        settings.h_left,
        real_observations,
    )
    obs = states_kf[settings.ilocs_waterlevel, :]
    # obs = real_observations[:, :-1]

    errors = []
    Ns = np.arange(5, 500, 10)
    for ensemble_size in tqdm(Ns):
        states_enkf, _ = ensemble_kalman_filter(
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
        estimated_observations = states_enkf[settings.ilocs_waterlevel, :]
        errors.append(np.linalg.norm(obs - estimated_observations))

    plt.plot(Ns, 5 / np.sqrt(Ns), "r", label="5/sqrt(N)")
    plt.plot(Ns, errors, label="Errors")
    plt.xlabel("No. Ensembles")
    plt.ylabel("Error")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(Plotter.add_folder("rate.pdf"), bbox_inches="tight")
    plt.show()
