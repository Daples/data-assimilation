import numpy as np
from typing import Any

from utils._typing import DynamicMatrix


def kalman_filter(
    M: DynamicMatrix,
    B: DynamicMatrix,
    C: DynamicMatrix,
    Q: DynamicMatrix,
    R: DynamicMatrix,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    measurements: list[Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Kalman fkkaskmdasmd.

    Parameters
    ----------
    A
        System matrix
    B
        Input
    C
        Output (measurements)
    Q
        System noise covariance
    R
        Measurement noise covariance
    """

    num_states = M(0).shape[0]

    state = initial_state
    covariance = initial_covariance

    estimated_states = []
    estimated_covariances = []

    for k, measurement in enumerate(measurements):
        # Predict step (forecast)
        predicted_state = M(k) @ state
        predicted_covariance = M(k) @ covariance @ M(k).T + Q(k)

        # Update step
        innovation = measurement - C(k) @ predicted_state
        innovation_covariance = C(k) @ predicted_covariance @ C(k).T + R(k)
        kalman_gain = (
            predicted_covariance @ C(k).T @ np.linalg.inv(innovation_covariance)
        )

        state = predicted_state + kalman_gain @ innovation
        covariance = (np.eye(num_states) - kalman_gain @ C(k)) @ predicted_covariance

        estimated_covariances.append(covariance)
        estimated_states.append(state)

    return estimated_states, estimated_covariances
