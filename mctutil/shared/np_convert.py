import numpy as np
import numpy.typing as npt


def np_convert(target_dtype: np.dtype, source: npt.ArrayLike, normalize=True, safe_bool=False):
	"""Convert array data into a target numpy dtype.

	Integer and floating-point targets normalize over the source range by default.
	"""
	target_dtype = np.dtype(target_dtype)
	source_array = np.asarray(source)

	if safe_bool and target_dtype == np.dtype(bool):
		return source_array.astype(target_dtype).astype(np.uint8)
	elif np.issubdtype(target_dtype, np.integer) and normalize:
		dtype_range = np.iinfo(target_dtype).max - np.iinfo(target_dtype).min
		source_floor = np.min(source_array) * -1
		source_range = np.max(source_array) + source_floor

		if source_range == 0.0:
			source_range = 1.0

		return ((source_array + source_floor) * max(dtype_range / source_range, 1)).astype(target_dtype)
	elif np.issubdtype(target_dtype, np.floating) and normalize:
		source_floor = np.min(source_array) * -1
		source_range = np.max(source_array) + source_floor

		if source_range == 0.0:
			source_range = 1.0

		return ((source_array + source_floor) / source_range).astype(target_dtype)
	else:
		return source_array.astype(target_dtype)
