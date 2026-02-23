# once we have our clone average saved, we need to be able to parse it for its covariance matrix
import numpy as np


def retrieve_C_S(L: int):
    # given the truncation degree, we can reconstruct the entire file name and put together the harmonics matrices
    file_name = f"/clone_averages/grgm1200a_clone_mean_L{L}.npz"
    data = np.load(file_name)
    C = data["C"]
    S = data["S"]
    return C, S