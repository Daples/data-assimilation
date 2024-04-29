import numpy as np

from utils._typing import DynamicMatrix


def kalman_filter(
    M: DynamicMatrix,
    B: DynamicMatrix,
    H: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    input_vector: np.ndarray,
    measurements: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Kalman fkkaskmdasmd.

    Parameters
    ----------
    M
        System matrix
    B
        Input
    H
        Output (measurements)
    Q
        System noise covariance
    R
        Measurement noise covariance
    initial_state

    initial_covariance

    input_vector

    measurements

    """

    num_states = M(0).shape[0]
    num_obs = H(0).shape[0]

    state = initial_state
    covariance = initial_covariance

    obs_time = measurements.shape[1] - 1
    estimated_states = np.zeros((num_states, obs_time))
    estimated_covariances = []

    for k in range(obs_time):
        measurement = measurements[:, k].reshape((num_obs, 1))
        # Predict step (forecast)
        predicted_state = M(k) @ state + B(k) * input_vector[k]
        predicted_covariance = M(k) @ covariance @ M(k).T + Q(k)

        # Update step
        innovation = measurement - H(k) @ predicted_state
        innovation_covariance = H(k) @ predicted_covariance @ H(k).T + R(k)
        kalman_gain = (
            predicted_covariance @ H(k).T @ np.linalg.inv(innovation_covariance)
        )

        state = predicted_state + kalman_gain @ innovation
        covariance = (np.eye(num_states) - kalman_gain @ H(k)) @ predicted_covariance

        estimated_covariances.append(covariance)
        estimated_states[:, k] = state.squeeze()

    return estimated_states, estimated_covariances
