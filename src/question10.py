import numpy as np
from scipy.sparse.linalg import inv
from scipy.sparse import bmat, csr_array
import utils.time_series as time_series
import scipy.sparse as sp

from utils.plotter import Plotter
from utils.simulate import forward
from tqdm import tqdm

from utils.settings import Settings
from filtering.kalman import kalman_filter

hours_to_seconds = 60.0 * 60.0
days_to_seconds = 24.0 * 60.0 * 60.0
figs_folder = "./figs"
data_file = lambda s: f"tide_{s}.txt"
storm_file = lambda s: f"waterlevel_{s}.txt"


def loopQ10() -> None:
    settings = Settings(add_noise=True)

    times = range(0, 25)

    rmse_list = np.zeros([4, len(times)])
    biases_list = np.zeros([4, len(times)])
    mae_list = np.zeros([4, len(times)])

    for i, lead_time in enumerate(times):
        rmse, biase, mae = question10(lead_time=lead_time)
        rmse_list[:, i] = rmse
        biases_list[:, i] = biase
        mae_list[:, i] = mae
    Plotter.plot_lts(times, biases_list, settings, "biases", "Biases")
    Plotter.plot_lts(times, rmse_list, settings, "rmses", "RMSE")
    Plotter.plot_lts(times, mae_list, settings, "mae", "MAE")


def question10(lead_time: int | None = None) -> tuple[list[float], ...]:
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
    obs_times, observed_data = time_series.read_datasets(datasets)

    # Indexing storm peak and forecasting
    lt = 8
    if lead_time is not None:
        lt = lead_time
    lt_conv = int(lt * 60 / 10)
    index_peak_storm = int(np.argmax(observed_data[0, :]))
    cut_index = index_peak_storm - lt_conv

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
        len(forecast_times),
        deterministic=True,
    )

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

    indices = [settings.ilocs_waterlevel[1]]
    # indices = settings.ilocs_waterlevel
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
            observed_data[:, :-1],
            settings,
            variable_name,
            shift=1,
            forecast=(cut_index, state_forecast),
            prefix=f"lt{lt}_forecast",
            # prefix=f"forecast",
        )

    estimated_observations = states[settings.ilocs_waterlevel, :]
    rmses, biases, maes = time_series.get_statistics(
        state_forecast, observed_data[:, cut_index - 1 :], settings, init_n=1
    )

    return rmses, biases, maes
