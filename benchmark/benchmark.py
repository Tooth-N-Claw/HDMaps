import os
import time
import numpy as np
import matplotlib.pyplot as plt

from HDM import hdm_embed, HDMConfig, compute_fiber_kernel_from_maps
from scipy.io import loadmat
from scipy.sparse import random
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize

from HDM.visualization_tools import embed_vs_actual

# --- NEW: imports for memory sampling ---
import psutil
import threading
import gc

def generate_samples(data_dim, data_points, num_samples, seed):
    np.random.seed(seed)
    data = [np.random.uniform(-1, 1, size=(data_points, data_dim)) for _ in range(num_samples)]
    return data

def generate_single_sparse_map(num_datapoints, density, seed):
    rng = np.random.RandomState(seed)
    sparse_map = random(num_datapoints, num_datapoints, density=density, format='coo', random_state=rng)

    # make it symmetric
    rows, cols = sparse_map.nonzero()
    mask = rows <= cols
    upper_rows = rows[mask]
    upper_cols = cols[mask]
    upper_data = sparse_map.data[mask]

    all_rows = np.concatenate([upper_rows, upper_cols])
    all_cols = np.concatenate([upper_cols, upper_rows])
    all_data = np.concatenate([upper_data, upper_data])

    return coo_matrix((all_data, (all_rows, all_cols)), shape=sparse_map.shape)

def generate_sparse_maps(num_samples, num_datapoints, density, seed, n_jobs=-1):
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31, size=num_samples * num_samples)

    sparse_matrices = Parallel(n_jobs=n_jobs)(
        delayed(generate_single_sparse_map)(num_datapoints, density, s) for s in seeds
    )

    map_matrix = np.empty((num_samples, num_samples), dtype=object)
    for idx, sparse_map in enumerate(sparse_matrices):
        map_matrix[idx // num_samples, idx % num_samples] = sparse_map

    return map_matrix

def visualize_corresponding_points(points: np.ndarray, num_samples) -> None:
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    scalars = np.tile(np.arange(6), num_samples)
    cmap = cm.get_cmap("rainbow", 6)
    norm = Normalize(vmin=0, vmax=6-1)

    plotter.add_mesh(point_cloud, scalars=scalars, point_size=10,
                     render_points_as_spheres=True, cmap="rainbow",
                     clim=[0, 5], show_scalar_bar=False)

    plotter.show()

def single_run(data_samples, fiber_kernel, config):
    hdm_embed(
        data_samples=data_samples,
        config=config,
        fiber_kernel=fiber_kernel
    )

def generate_data(sample_size, data_point_amount, data_dim, seed, density):
    data_samples = generate_samples(data_dim, data_point_amount, sample_size, seed)
    maps = generate_sparse_maps(sample_size, data_point_amount, density, seed)
    fiber_kernel = compute_fiber_kernel_from_maps(maps)
    return data_samples, fiber_kernel

def time_single_run(data_samples, fiber_kernel, config) -> float:
    t0 = time.perf_counter()
    single_run(data_samples, fiber_kernel, config)
    return time.perf_counter() - t0

# --- NEW: helpers to measure peak RAM during the run ---
def _current_rss_bytes(include_children: bool = True) -> int:
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    if include_children:
        for c in p.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.Error:
                pass
    return rss

def run_and_measure(data_samples, fiber_kernel, config, sample_interval: float = 0.02):
    """
    Run one embed; return (elapsed_seconds, peak_extra_MiB).
    peak_extra_MiB = max(RSS during run) - baseline RSS before run.
    """
    gc.collect()
    baseline = _current_rss_bytes()
    stop = threading.Event()
    peak = {"bytes": baseline}

    def sampler():
        while not stop.is_set():
            cur = _current_rss_bytes()
            if cur > peak["bytes"]:
                peak["bytes"] = cur
            time.sleep(sample_interval)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        single_run(data_samples, fiber_kernel, config)
    finally:
        stop.set()
        t.join()
    elapsed = time.perf_counter() - t0
    peak_extra_mib = max(0.0, (peak["bytes"] - baseline) / (1024 ** 2))
    return elapsed, peak_extra_mib

def run_benchmark(save_path: str = None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    sample_sizes = [2**i for i in range(3, 9)]
    data_point_amounts = [2**i for i in range(3, 9)]

    config = HDMConfig(
        base_sparsity=0.08,
        fiber_sparsity=0.08,
        device="cpu"
    )

    repetitions = 3
    seed = 42
    data_dim = 3
    density = 1.0

    fixed_data_point_amount = 100
    fixed_sample_size = 100

    times_vs_ss = []
    times_vs_dpa = []
    
    mem_vs_ss = []
    mem_vs_dpa = []

    for sample_size in sample_sizes:
        print(f"sampel size: {sample_size}")
        data_samples, fiber_kernel = generate_data(sample_size, fixed_data_point_amount, data_dim, seed, density)
        # warmup
        single_run(data_samples, fiber_kernel, config)

        times, mems = [], []
        for _ in range(repetitions):
            t_sec, m_mib = run_and_measure(data_samples, fiber_kernel, config)
            times.append(t_sec)
            mems.append(m_mib)
        times_vs_ss.append(np.mean(times))
        mem_vs_ss.append(np.mean(mems))

    for data_point_amount in data_point_amounts:
        print(f"data point amount: {data_point_amount}")
        data_samples, fiber_kernel = generate_data(fixed_sample_size, data_point_amount, data_dim, seed, density)
        # warmup
        single_run(data_samples, fiber_kernel, config)

        times, mems = [], []
        for _ in range(repetitions):
            t_sec, m_mib = run_and_measure(data_samples, fiber_kernel, config)
            times.append(t_sec)
            mems.append(m_mib)
        times_vs_dpa.append(np.mean(times))
        mem_vs_dpa.append(np.mean(mems))

    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.set_xlabel("sample_size")
    ax1.set_ylabel("avg time per run [s]", color=color)
    ax1.plot(sample_sizes, times_vs_ss, marker='o', color=color, label="Runtime")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("avg peak extra memory [MiB]", color=color)
    ax2.plot(sample_sizes, mem_vs_ss, marker='s', linestyle="--", color=color, label="Memory")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Runtime & Memory vs sample_size (data_point_amount={fixed_data_point_amount})')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "runtime_mem_vs_sample_size.png"))
    plt.close()

    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.set_xlabel("data_point_amount")
    ax1.set_ylabel("avg time per run [s]", color=color)
    ax1.plot(data_point_amounts, times_vs_dpa, marker='o', color=color, label="Runtime")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("avg peak extra memory [MiB]", color=color)
    ax2.plot(data_point_amounts, mem_vs_dpa, marker='s', linestyle="--", color=color, label="Memory")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Runtime & Memory vs data_point_amount (sample_size={fixed_sample_size})')
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "runtime_mem_vs_data_point_amount.png"))
    plt.close()


if __name__ == "__main__":
    run_benchmark(save_path="benchmark/plots")
