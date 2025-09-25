import os
import time
import numpy as np
import matplotlib.pyplot as plt

from HDM import hdm_embed, HDMConfig, compute_fiber_kernel_from_maps
from scipy.sparse import coo_matrix
from scipy import sparse
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize


import psutil
import threading
import gc

from load_off import generate


def generate_samples(data_dim, data_points, num_samples, seed):
    np.random.seed(seed)
    data = [
        np.random.uniform(-1, 1, size=(data_points, data_dim))
        for _ in range(num_samples)
    ]
    return data


def generate_sparse_maps(num_samples, num_datapoints, amount):
    map_matrix = np.empty((num_samples, num_samples), dtype=object)
    for i in range(num_samples * num_samples):
        rows = np.random.randint(0, num_datapoints, size=amount)
        cols = np.random.randint(0, num_datapoints, size=amount)
        data = np.random.random(amount)
        sparse_map = coo_matrix(
            (data, (rows, cols)), shape=(num_datapoints, num_datapoints)
        )

        row_sums = np.asarray(sparse_map.sum(axis=1)).ravel()
        inv = np.zeros_like(row_sums)
        nz = row_sums > 0
        inv[nz] = 1.0 / row_sums[nz]
        Dinv = sparse.diags(inv)
        if i // num_samples == i % num_samples:
            map_matrix[i // num_samples, i % num_samples] = sparse.identity(
                num_datapoints, format="coo"
            )
        else:
            map_matrix[i // num_samples, i % num_samples] = Dinv @ sparse_map

    return map_matrix


def visualize_corresponding_points(points: np.ndarray, num_samples) -> None:
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter()

    scalars = np.tile(np.arange(6), num_samples)
    cmap = cm.get_cmap("rainbow", 6)
    norm = Normalize(vmin=0, vmax=6 - 1)

    plotter.add_mesh(
        point_cloud,
        scalars=scalars,
        point_size=10,
        render_points_as_spheres=True,
        cmap="rainbow",
        clim=[0, 5],
        show_scalar_bar=False,
    )

    plotter.show()


def single_run(data_samples, fiber_kernel, config):
    return hdm_embed(
        data_samples=data_samples,
        # base_distances=base_distances,
        # block_indices=block_indices,
        config=config,
        fiber_kernel=fiber_kernel,
    )


def generate_data(sample_size, data_point_amount, data_dim, seed, amount):
    data_samples = generate(
        sample_size,
        data_point_amount,
    )
    # data_samples = load_teeths()

    print("generated data samples")
    maps = generate_sparse_maps(sample_size, data_point_amount, amount)
    # maps = loadmat("examples/platyrrhine/softMapMatrix.mat")["softMapMatrix"]

    fiber_kernel = compute_fiber_kernel_from_maps(maps)
    print("converted map to fiber kernel")
    return data_samples, fiber_kernel


def time_single_run(data_samples, fiber_kernel, config) -> float:
    t0 = time.perf_counter()
    single_run(data_samples, fiber_kernel, config)
    return time.perf_counter() - t0


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
    peak_extra_mib = max(0.0, (peak["bytes"] - baseline) / (1024**2))
    return elapsed, peak_extra_mib


def run_benchmark(save_path: str = None):
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # sample_sizes = [2**i for i in range(10, 11)]
    sample_sizes = [50]
    data_point_amounts = [2**i for i in range(5, 6)]

    config = HDMConfig(
        base_sparsity=0.4,
        # base_sparsity=0.08,
        # fiber_sparsity=0.08,
        device="gpu",
    )

    repetitions = 3
    seed = 42
    data_dim = 3
    density = 0.03

    fixed_data_point_amount = 4463
    fixed_sample_size = 100

    times_vs_ss = []
    times_vs_dpa = []

    mem_vs_ss = []
    mem_vs_dpa = []

    for sample_size in sample_sizes:
        print(f"sampel size: {sample_size}")
        data_samples, fiber_kernel = generate_data(
            sample_size, fixed_data_point_amount, data_dim, seed, 50000
        )
        # warmup
        points = single_run(data_samples, fiber_kernel, config)
        points = pv.PolyData(points)
        plotter = pv.Plotter()
        plotter.add_mesh(
            points, color="red", point_size=5, render_points_as_spheres=True
        )
        plotter.show()

        times, mems = [], []
        for _ in range(repetitions):
            t_sec, m_mib = run_and_measure(data_samples, fiber_kernel, config)
            times.append(t_sec)
            mems.append(m_mib)
        times_vs_ss.append(np.mean(times))
        mem_vs_ss.append(np.mean(mems))
    exit(0)
    for data_point_amount in data_point_amounts:
        print(f"data point amount: {data_point_amount}")
        data_samples, fiber_kernel = generate_data(
            fixed_sample_size, data_point_amount, data_dim, seed, 5000
        )
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
    ax1.plot(sample_sizes, times_vs_ss, marker="o", color=color, label="Runtime")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("avg peak extra memory [MiB]", color=color)
    ax2.plot(
        sample_sizes, mem_vs_ss, marker="s", linestyle="--", color=color, label="Memory"
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(
        f"Runtime & Memory vs sample_size (data_point_amount={fixed_data_point_amount})"
    )
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "runtime_mem_vs_sample_size.png"))
    plt.close()

    fig, ax1 = plt.subplots()
    color = "tab:blue"
    ax1.set_xlabel("data_point_amount")
    ax1.set_ylabel("avg time per run [s]", color=color)
    ax1.plot(data_point_amounts, times_vs_dpa, marker="o", color=color, label="Runtime")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("avg peak extra memory [MiB]", color=color)
    ax2.plot(
        data_point_amounts,
        mem_vs_dpa,
        marker="s",
        linestyle="--",
        color=color,
        label="Memory",
    )
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(
        f"Runtime & Memory vs data_point_amount (sample_size={fixed_sample_size})"
    )
    fig.tight_layout()
    plt.savefig(os.path.join(save_path, "runtime_mem_vs_data_point_amount.png"))
    plt.close()


if __name__ == "__main__":
    run_benchmark(save_path="benchmark/plots")
