from typing import Callable
import numpy as np
from scipy.sparse import csr_array

DynamicMatrix = Callable[[int | float], np.ndarray | csr_array]
DataArray = list[float | int] | np.ndarray
