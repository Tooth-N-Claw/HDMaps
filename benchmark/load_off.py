import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist


def load_off_vertices(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
    if lines[0] != "OFF":
        raise ValueError(f"{path} does not start with OFF")
    n_vertices, _, _ = map(int, lines[1].split())
    return np.array(
        [list(map(float, lines[i].split())) for i in range(2, 2 + n_vertices)],
        dtype=float,
    )


def load_all_off_vertices(folder):
    folder = Path(folder)
    arrays = [load_off_vertices(p) for p in sorted(folder.glob("*.off"))]
    return np.vstack(arrays)


def generate(data_samples, data_points, seed=None):
    all_vertices = load_all_off_vertices("examples/platyrrhine/ReparametrizedOFF")
    N = all_vertices.shape[0]
    rng = np.random.default_rng(seed)

    idx = rng.integers(N, size=(data_samples, data_points))
    samples = all_vertices[idx]
    return normalize_data_samples(samples)


def load_teeths():
    def load_all_off_vertices(folder):
        folder = Path(folder)
        arrays = [load_off_vertices(p) for p in sorted(folder.glob("*.off"))]
        return arrays

    return normalize_data_samples(
        load_all_off_vertices("examples/platyrrhine/ReparametrizedOFF")
    )


def normalize_data_samples(data_samples):
    for i in range(len(data_samples)):
        max_dist = pdist(data_samples[i]).max()
        sample_min = data_samples[i].min(axis=0)
        data_samples[i] = (data_samples[i] - sample_min) / max_dist
    return data_samples
