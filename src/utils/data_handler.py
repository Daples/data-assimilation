import numpy as np

from utils._typing import DataArray


class DataHandler:
    """A class to homogenize the management and typing of data."""

    @classmethod
    def __cast_array__(cls, *args: DataArray) -> tuple[np.ndarray, ...]:
        """It converts the input array types into the standardized type.

        Parameters
        ----------
        args: utils._typing.DataArray
            The arrays to convert.

        Returns
        -------
        tuple[utils._typing.DataArrray, ...]
            The transformed arrays.
        """

        return tuple(map(lambda x: np.array(x), args))
