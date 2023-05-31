import numpy as np
from utils._typing import DynamicMatrix
from utils.settings import Settings
from scipy.sparse.linalg import spsolve
from utils.plotter import Plotter
from tqdm import tqdm


def timestep(x: np.ndarray, i: int, settings: Settings) -> np.ndarray:
    # return (h,u) one timestep later
    temp = x.copy()
    A = settings.A
    B = settings.B
    rhs = B.dot(temp)  # B*x
    rhs[0] = settings.h_left[i] + settings.forcing_noise[i]  # left boundary
    newx = spsolve(A, rhs)
    return newx


def simulate_real(
    settings: Settings, plot: bool = False, prefix: str = "ffig_map"
) -> tuple[np.ndarray, np.ndarray]:
    x, _ = settings.initialize()
    ts = settings.ts[:]

    observation_series = np.zeros((len(settings.ilocs), len(ts)))
    complete_series = np.zeros((x.shape[0], len(ts)))
    for i in tqdm(np.arange(1, len(ts))):
        x = timestep(x, i, settings)
        if plot:
            Plotter.plot_state(x, i, settings, name=prefix)
        complete_series[:, i] = x
        observation_series[:, i] = x[settings.ilocs]
    return complete_series, observation_series


def forward(
    M: DynamicMatrix,
    B: DynamicMatrix,
    H: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state: np.ndarray,
    input_vector: np.ndarray,
    n_steps: int,
    ws: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (states, observations) for n_steps ahead (steps state vectors
    including the initial condition)"""

    n_x = M(0).shape[0]
    n_z = H(0).shape[0]
    states = np.zeros((n_x, n_steps))
    observations = np.zeros((n_z, n_steps))

    # Noise handles
    if ws is None:
        w = lambda t: np.random.multivariate_normal(np.zeros(n_x), Q(t - 1)).reshape(
            (n_x, 1)
        )
    else:
        aux = np.zeros((n_x - 1, 1))
        w = lambda t: np.vstack((aux, ws[t]))
    v = lambda t: np.random.multivariate_normal(np.zeros(n_z), R(t)).reshape((n_z, 1))

    # Initial states
    states[:, 0] = initial_state.squeeze()
    initial_observations = H(0) @ initial_state + v(0)
    observations[:, 0] = initial_observations.squeeze()

    for t in range(1, n_steps):
        x = states[:, t - 1].reshape((n_x, 1))
        states[:, t] = (M(t) @ x + B(t) * input_vector[t - 1] + w(t - 1)).squeeze()
        observations[:, t] = (H(t) @ (states[:, t]).reshape((n_x, 1)) + v(t)).squeeze()

    return states, observations
