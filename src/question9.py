import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
import utils.time_series as time_series
import scipy.sparse as sp

from utils.plotter import Plotter
from tqdm import tqdm

from utils.settings import Settings
from utils.simulate import simulate_real
from filtering.kalman import kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def question9() -> None:
    """"""

    settings = Settings(add_noise=True)
    settings.initialize()

    states_model, obs_model = simulate_real(settings) 


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
    initial_covariance = 0.01 * np.eye(n_state)

    # Load observations
    station_names = list(map(lambda s: s.lower(), settings.names[1:]))
    datasets = list(map(storm_file, station_names))
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

    times = [5, 50, 100]
    for t in tqdm(times):
        Plotter.plot_KF_states(
            t,
            states,
            covariances,
            observed_data,
            settings,
            stations_shift=1,
            prefix="storm_",
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
            observed_data,
            settings,
            variable_name,
            shift=1,
            prefix="storm_",
            real= states_model
        )

    estimated_observations = states[settings.ilocs_waterlevel, :]
    rmses, biases, mae = time_series.get_statistics(
        estimated_observations, observed_data, settings, init_n=1
    )

    output = np.array([biases, rmses]).T
    np.savetxt("table_question9.csv", output, delimiter=",", fmt="%1.4f")

    rmses, biases, mae = time_series.get_statistics(
        obs_model, observed_data, settings, init_n=1
    )

    output = np.array([biases, rmses]).T
    np.savetxt("table_question9_model.csv", output, delimiter=",", fmt="%1.4f")
    print(obs_model)
    print(observed_data.shape)
    nb_sections = 4
    for t in range(nb_sections):
        end_min = np.min([(t+1)*obs_model.shape[1]//nb_sections,obs_model.shape[1] ])
        sub_obs_model = obs_model[:, t*obs_model.shape[1]//nb_sections:end_min]
        sub_obs_estime = estimated_observations[:, t*obs_model.shape[1]//nb_sections:end_min]
        sub_obs_data = observed_data[:, t*obs_model.shape[1]//nb_sections:end_min+1]

        rmses, biases, mae = time_series.get_statistics(
        sub_obs_estime, sub_obs_data, settings, init_n=1
        )
        print(t)
        print(rmses)
        print(biases)
        print(mae)

