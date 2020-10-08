import numpy as np


def min_max_normalization(x: np.ndarray) -> np.ndarray:
    """ Min max normalization
    """
    xt = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    return xt


def proportion(x: np.ndarray) -> np.ndarray:
    """ Proportion of sum
    """
    return x / np.sum(x)


def normalize_and_proportion(x: np.ndarray) -> np.ndarray:
    """ Min max normalization followed by proportion
    """
    return proportion(min_max_normalization(x))


def dict_subset_by_key(x: dict, keys: list) -> dict:
    new_dict = {k: x[k] for k in keys}

    return new_dict
