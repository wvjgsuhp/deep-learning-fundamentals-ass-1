import numpy as np
import numpy.typing as npt
import pandera.typing as pa

NPFloats = npt.NDArray[np.float64]
NPFloatMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
NPInt = npt.NDArray[np.int64]
NPIntMatrix = np.ndarray[tuple[int, int], np.dtype[np.int64]]
PDFloats = pa.Series[float]
