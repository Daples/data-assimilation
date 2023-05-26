from typing import Callable
import numpy as np

DynamicMatrix = Callable[[int | float], np.ndarray]
