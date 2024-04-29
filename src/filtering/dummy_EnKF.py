from typing import Callable

import numpy as np

DynamicMatrix = Callable[[float | int], np.ndarray]


def ensemble_kalman_filter(
    M: DynamicMatrix,
    H: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state_mean: np.ndarray,
    initial_covariance: np.ndarray,
    measurements: np.ndarray,
    ensemble_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """A standard implementation of the EnKF."""

    num_states = M(0).shape[0]
    num_obs = H(0).shape[0]

    state_ensemble = np.random.multivariate_normal(
        initial_state_mean.flatten(), initial_covariance, ensemble_size
    ).T

    obs_time = measurements.shape[1] - 1
    estimated_states = np.zeros((num_states, obs_time))
    estimated_covariances = []

    for k in range(obs_time):
        measurement = measurements[:, k].reshape((num_obs, 1))

        # Predict step
        w = np.random.multivariate_normal(np.zeros(num_states), Q(k), ensemble_size).T
        predicted_state_ensemble = M(k) @ state_ensemble + w

        P = np.cov(predicted_state_ensemble, ddof=1)

        v = np.random.multivariate_normal(np.zeros(num_obs), R(k), ensemble_size).T
        innovations = measurement - H(k) @ predicted_state_ensemble - v
        innovation_covariance = H(k) @ P @ H(k).T + R(k)
        inv_innovation_covariance = np.linalg.inv(innovation_covariance)

        kalman_gain = P @ H(k).T @ inv_innovation_covariance
        state_ensemble = predicted_state_ensemble + kalman_gain @ innovations

        updated_state_mean = np.mean(state_ensemble, axis=1).reshape((num_states, 1))
        updated_state_covariance = np.cov(state_ensemble, ddof=1)

        estimated_states[:, k] = updated_state_mean.squeeze()
        estimated_covariances.append(updated_state_covariance)

    return estimated_states, estimated_covariances
