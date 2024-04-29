import numpy as np
from utils._typing import DynamicMatrix


def ensemble_kalman_filter(
    M: DynamicMatrix,
    B: DynamicMatrix,
    H: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state_mean: np.ndarray,
    initial_covariance: np.ndarray,
    input_vector: np.ndarray,
    measurements: np.ndarray,
    ensemble_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """"""

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
        u = input_vector[k] * np.ones((1, ensemble_size))
        predicted_state_ensemble = M(k) @ state_ensemble + B(k) @ u + w

        predicted_state = predicted_state_ensemble.mean(axis=1).reshape((num_states, 1))

        # "Efficient formulation" squared root algorithm
        L = (
            1
            / np.sqrt(ensemble_size - 1)
            * (predicted_state_ensemble - predicted_state)
        )
        Y = H(k) @ L

        v = np.random.multivariate_normal(np.zeros(num_obs), R(k), ensemble_size).T
        innovations = measurement - H(k) @ predicted_state_ensemble - v
        innovation_covariance = Y @ Y.T + R(k)
        inv_innovation_covariance = np.linalg.inv(innovation_covariance)

        kalman_gain = L @ Y.T @ inv_innovation_covariance
        state_ensemble = predicted_state_ensemble + kalman_gain @ innovations

        updated_state_mean = np.mean(state_ensemble, axis=1).reshape((num_states, 1))
        # updated_state_covariance = (
        #     L @ (np.eye(ensemble_size) - Y.T @ inv_innovation_covariance @ Y) @ L.T
        # )
        updated_state_covariance = np.cov(state_ensemble, ddof=1)

        estimated_states[:, k] = updated_state_mean.squeeze()
        estimated_covariances.append(updated_state_covariance)

    return estimated_states, estimated_covariances


def ensemble_kalman_filter_st(
    M: DynamicMatrix,
    B: DynamicMatrix,
    H: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state_mean: np.ndarray,
    initial_covariance: np.ndarray,
    input_vector: np.ndarray,
    measurements: np.ndarray,
    ensemble_size: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """"""

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
        u = input_vector[k] * np.ones((1, ensemble_size))
        predicted_state_ensemble = M(k) @ state_ensemble + B(k) @ u + w

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
