


import os

import numpy as np


def get_wing_data(samples, data_points):
    path = "examples/example-data/wing"
    files = [f for f in os.listdir(path) if f.endswith(".txt")]
    files = files[:samples]
    data_samples = [
        np.loadtxt(os.path.join(path, file), delimiter=",")[:data_points] for file in files
    ]   
    return data_samples