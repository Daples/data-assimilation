import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
import matplotlib.pyplot as plt

from utils.plotter import Plotter
from utils.simulate import forward, simulate_real

from utils.settings import Settings


def compare_representations() -> None:
    settings = Settings(add_noise=True)
    x0, _ = settings.initialize()
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
    _M = lambda _: M.toarray()
    _B = lambda _: B.toarray()
    _H = lambda _: H.toarray()
    _Q = lambda _: Q.toarray()
    _R = lambda _: R.toarray()

    init_state = np.hstack((x0, 0)).reshape((n_state, 1))
    input_signal = settings.h_left
    steps = len(settings.ts)

    states, observations = forward(
        _M, _B, _H, _Q, _R, init_state, input_signal, steps, ws=settings.white_noise
    )
    _, observations_real = simulate_real(settings)

    k = -1
    plt.plot(settings.ts, states[k, :], "r", label="State space")
    # plt.plot(settings.ts, observations_real[k, :], "k", label="Simulation")
    plt.plot(settings.ts, settings.forcing_noise, "k", label="Simulation")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    Plotter.__setup_config__()
    compare_representations()
