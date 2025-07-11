import numpy as np
from .utils import HDMConfig


METRICS = {"frobenius", "euclidean"}


def validate_config(config: HDMConfig):
    if config.base_metric not in METRICS:
        raise ValidationError(f"{config.base_metric} is not a valid base metric.")

    if config.fiber_metric not in METRICS:

        raise ValidationError(f"{config.fiber_metric} is not a valid fiber metric.")


def validate_data(data_samples: list[np.ndarray]):
    if not isinstance(data_samples, list):
        raise ValidationError("data_samples must be a list")

    if not all(isinstance(arr, np.ndarray) for arr in data_samples):
        raise ValidationError("All items in data_samples must be NumPy arrays")

    if len(data_samples) == 0:
        raise ValidationError("data_samples cannot be empty")
