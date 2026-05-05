"""
Sparse B SpMM Benchmarking Suite (MPI HPC)

Mirrors the OpenMP HPC benchmark structure, but evaluates MPI process scaling
using row-wise distributed-memory partitioning of A.

Usage:
    python benchmark_spmm_sparse_mpi.py

Sample usage:
    python benchmark_spmm_sparse_mpi.py --outdir results_hpc_spmm_mpi --processes 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "matrix_matrix_mult"))
sys.path.insert(0, BASE_DIR)

from benchmark_spmm_sparse_serial import generate_sparse_b, load_local_matrix  # noqa: E402
from spmm_mpi import benchmark_spmm_algorithm_sparse_b_mpi  # noqa: E402


HPC_DEFAULT_MATRICES = [
    "1138_bus",
    "abb313",
    "delaunay_n15",
    "bcsstk30",
    "delaunay_n19",
    "pkustk14",
]


def build_worker_launch_cmd(
    args: argparse.Namespace,
    num_procs: int,
    worker_csv: Path,
) -> list[str]:
    """Build one MPI worker launch command for a given process count."""
    launcher_name = os.path.basename(args.mpi_launcher)
    python_executable = str(Path(sys.executable).resolve())
    script_path = str(Path(__file__).resolve())

    cmd = [args.mpi_launcher]
    if launcher_name == "srun":
        # Nested srun job steps are more reliable when they explicitly overlap the
        # launcher step and inherit the current environment.
        cmd.extend(["--overlap", "--export=ALL", "--ntasks", str(num_procs)])
    else:
        cmd.extend(["-np", str(num_procs)])

    cmd.extend(
        [
            python_executable,
            script_path,
            "--mode",
            "worker",
            "--output",
            str(worker_csv),
            "--outdir",
            args.outdir,
            "--cache-dir",
            args.cache_dir,
            "--repeats",
            str(args.repeats),
        ]
    )
    if args.validate:
        cmd.append("--validate")
    if args.matrices:
        cmd.append("--matrices")
        cmd.extend(args.matrices)
    if args.b_cols:
        cmd.append("--b-cols")
        cmd.extend(str(v) for v in args.b_cols)
    if args.sparsity:
        cmd.append("--sparsity")
        cmd.extend(str(v) for v in args.sparsity)
    return cmd


def validate_worker_launch_prereqs() -> None:
    """Fail early with a clearer message if the worker executable path is unusable."""
    python_path = Path(sys.executable).resolve()
    script_path = Path(__file__).resolve()

    if not python_path.exists():
        raise FileNotFoundError(f"Python executable not found: {python_path}")
    if not os.access(python_path, os.X_OK):
        raise PermissionError(f"Python executable is not runnable: {python_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Worker script not found: {script_path}")


def benchmark_worker(
    cache_dir: str,
    matrix_names: list[str],
    b_cols: list[int],
    sparsities: list[float],
    repeats: int,
    output_csv: str,
    validate: bool = False,
) -> None:
    """Worker mode: run under mpirun for one fixed MPI process count."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    all_results: list[dict[str, float]] = []

    if rank == 0:
        print("=" * 90)
        print("Sparse B SpMM MPI Benchmark Worker")
        print("=" * 90)
        print(f"Matrices: {matrix_names}")
        print(f"B columns: {b_cols}")
        print(f"B sparsity: {sparsities}")
        print(f"MPI processes: {size}")
        print(f"Repeats: {repeats}")
        print("=" * 90)

    for matrix_name in matrix_names:
        if rank == 0:
            A, source = load_local_matrix(matrix_name, cache_dir)
            if A is None:
                print(f"✗ Could not load {matrix_name}, skipping...")
                matrix_payload = {"ok": False, "matrix_name": matrix_name}
            else:
                print(f"✓ Loaded {matrix_name} from {source}")
                matrix_payload = {"ok": True, "matrix_name": matrix_name}
        else:
            A = None
            matrix_payload = None

        matrix_payload = comm.bcast(matrix_payload, root=0)
        if not matrix_payload["ok"]:
            continue

        if rank == 0:
            m, n = A.shape
            print(f"\nBenchmarking {matrix_name} ({m}x{n}, nnz={A.nnz}):")
            print(f"  Density: {A.nnz / (m * n):.2e}")

        for k in b_cols:
            for sparsity in sparsities:
                if rank == 0:
                    print(f"\n  B: {A.shape[1]}x{k}, sparsity={sparsity:.2%} (nnz={int(A.shape[1] * k * sparsity)})")
                    B = generate_sparse_b(A.shape[1], k, sparsity=sparsity, dtype=A.dtype)
                else:
                    B = None

                for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
                    if rank == 0:
                        print(f"    Benchmarking {algo}...", end=" ", flush=True)
                    metrics = benchmark_spmm_algorithm_sparse_b_mpi(
                        A,
                        B,
                        algorithm=algo,
                        repeats=repeats,
                        validate=validate,
                    )
                    if rank == 0 and metrics is not None:
                        print(f"✓ ({metrics['gflops']:.2f} GFlop/s)")
                        all_results.append(
                            {
                                "matrix_name": matrix_name,
                                "m": A.shape[0],
                                "n": A.shape[1],
                                "nnz_a": metrics["nnz_a"],
                                "density_a": A.nnz / (A.shape[0] * A.shape[1]),
                                "k": k,
                                "nnz_b": metrics["nnz_b"],
                                "sparsity_b": sparsity,
                                "num_procs": metrics["num_procs"],
                                "algorithm": algo,
                                "nnz_c": metrics["nnz_c"],
                                "mean_time_sec": metrics["mean_time"],
                                "std_time_sec": metrics["std_time"],
                                "min_time_sec": metrics["min_time"],
                                "gflops": metrics["gflops"],
                                "memory_a_mb_total": metrics["memory_a_mb_total"],
                                "memory_b_mb_per_rank": metrics["memory_b_mb_per_rank"],
                                "memory_c_mb_total": metrics["memory_c_mb_total"],
                                "validation_ok": metrics["validation_ok"],
                            }
                        )

    if rank == 0:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        field_order = [
            "matrix_name",
            "m",
            "n",
            "nnz_a",
            "density_a",
            "k",
            "nnz_b",
            "sparsity_b",
            "num_procs",
            "algorithm",
            "nnz_c",
            "mean_time_sec",
            "std_time_sec",
            "min_time_sec",
            "gflops",
            "memory_a_mb_total",
            "memory_b_mb_per_rank",
            "memory_c_mb_total",
            "validation_ok",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Worker results written to {output_csv}")


def launch_worker_runs(args: argparse.Namespace) -> pd.DataFrame:
    """Launcher mode: mirror the OpenMP benchmark by looping over process counts."""
    validate_worker_launch_prereqs()
    os.makedirs(args.outdir, exist_ok=True)
    worker_dir = Path(args.outdir) / "_mpi_worker_runs"
    worker_dir.mkdir(exist_ok=True)

    print("=" * 90)
    print("Sparse B SpMM MPI Benchmark Launcher")
    print("=" * 90)
    print(f"Matrices: {args.matrices}")
    print(f"B columns: {args.b_cols}")
    print(f"B sparsity: {args.sparsity}")
    print(f"MPI process counts: {args.processes}")
    print(f"Repeats: {args.repeats}")
    print(f"Output directory: {args.outdir}")
    print("=" * 90)

    worker_csvs: list[Path] = []

    for num_procs in args.processes:
        worker_csv = worker_dir / f"worker_np{num_procs}.csv"
        cmd = build_worker_launch_cmd(args, num_procs, worker_csv)

        print(f"\nLaunching MPI run with {num_procs} processes...")
        print("Command:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        worker_csvs.append(worker_csv)

    frames = [pd.read_csv(path) for path in worker_csvs if path.exists()]
    if not frames:
        raise RuntimeError("No worker CSV outputs were produced")

    df = pd.concat(frames, ignore_index=True)
    final_csv = Path(args.outdir) / args.output
    df.to_csv(final_csv, index=False)
    print(f"\n✓ Combined results written to {final_csv}")

    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        plot_results(df, args.outdir)
    else:
        print("\n⚠ matplotlib not available—skipping plots")

    return df


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    """Mirror the OpenMP HPC plots, replacing threads with MPI process counts."""
    os.makedirs(output_dir, exist_ok=True)
    sparsity_label = ", ".join(f"{value:.0%}" for value in sorted(df["sparsity_b"].unique()))

    colors = {
        "row-wise": "#1f77b4",
        "outer-product": "#ff7f0e",
        "blocked": "#2ca02c",
        "scipy": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Sparse B SpMM MPI: Algorithm Comparison\nB sparsity: {sparsity_label}",
        fontsize=14,
        fontweight="bold",
    )

    matrices = sorted(df["matrix_name"].unique())
    x = np.arange(len(matrices))
    width = 0.2

    ax = axes[0, 0]
    for i, algo in enumerate(["row-wise", "outer-product", "blocked", "scipy"]):
        algo_times = [
            df[(df["matrix_name"] == matrix_name) & (df["algorithm"] == algo)]["mean_time_sec"].mean()
            for matrix_name in matrices
        ]
        ax.bar(x + (i - 1.5) * width, algo_times, width, label=algo, color=colors[algo])
    ax.set_xlabel("Matrix")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Execution Time by Matrix, Averaged over B columns and MPI process counts")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax = axes[0, 1]
    for i, algo in enumerate(["row-wise", "outer-product", "blocked", "scipy"]):
        algo_gflops = [
            df[(df["matrix_name"] == matrix_name) & (df["algorithm"] == algo)]["gflops"].mean()
            for matrix_name in matrices
        ]
        ax.bar(x + (i - 1.5) * width, algo_gflops, width, label=algo, color=colors[algo])
    ax.set_xlabel("Matrix")
    ax.set_ylabel("GFlop/s")
    ax.set_title("Performance by Matrix, Averaged over B columns and MPI process counts")
    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()

    ax = axes[1, 0]
    for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
        algo_df = df[df["algorithm"] == algo]
        nnz_perf = algo_df.groupby("nnz_b")["mean_time_sec"].mean().sort_index()
        ax.plot(
            nnz_perf.index,
            nnz_perf.values,
            label=algo,
            color=colors[algo],
            alpha=0.85,
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("nnz(B)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title("Execution Time vs B Sparsity")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    largest_matrix = (
        df[["matrix_name", "nnz_a"]]
        .drop_duplicates()
        .sort_values(["nnz_a", "matrix_name"])
        .iloc[-1]["matrix_name"]
    )
    ax = axes[1, 1]
    largest_df = df[df["matrix_name"] == largest_matrix]
    for algo in ["row-wise", "outer-product", "blocked", "scipy"]:
        algo_df = largest_df[largest_df["algorithm"] == algo]
        proc_perf = algo_df.groupby("num_procs")["mean_time_sec"].mean().sort_index()
        ax.plot(
            proc_perf.index,
            proc_perf.values,
            label=algo,
            color=colors[algo],
            marker="o",
            linewidth=2,
        )
    ax.set_xlabel("Number of Processes")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title(f"Process Scaling on {largest_matrix}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spmm_sparse_mpi_algorithm_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Sparse B SpMM MPI: B Column Count and Performance\nB sparsity: {sparsity_label}",
        fontsize=14,
        fontweight="bold",
    )

    max_procs = int(df["num_procs"].max())
    max_proc_df = df[df["num_procs"] == max_procs]

    ax = axes[0]
    for matrix_name in sorted(max_proc_df["matrix_name"].unique()):
        matrix_df = max_proc_df[
            (max_proc_df["matrix_name"] == matrix_name)
            & (max_proc_df["algorithm"] == "scipy")
        ]
        col_perf = matrix_df.groupby("k")["mean_time_sec"].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker="o", label=matrix_name, linewidth=2)
    ax.set_xlabel("B Column Count (k)")
    ax.set_ylabel("Mean Time (seconds)")
    ax.set_title(f"Execution Time vs B Columns (SciPy, {max_procs} processes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for matrix_name in sorted(max_proc_df["matrix_name"].unique()):
        matrix_df = max_proc_df[
            (max_proc_df["matrix_name"] == matrix_name)
            & (max_proc_df["algorithm"] == "scipy")
        ]
        col_perf = matrix_df.groupby("k")["gflops"].mean().sort_index()
        ax.plot(col_perf.index, col_perf.values, marker="s", label=matrix_name, linewidth=2)
    ax.set_xlabel("B Column Count (k)")
    ax.set_ylabel("GFlop/s")
    ax.set_title(f"Performance vs B Columns (SciPy, {max_procs} processes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "spmm_sparse_mpi_scaling.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Saved: {plot_path}")
    plt.close()

    for matrix_name in matrices:
        matrix_df = df[df["matrix_name"] == matrix_name]
        matrix_meta = matrix_df[["m", "n", "nnz_a"]].drop_duplicates().iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            (
                f"{matrix_name}: Runtime vs Processes by B Column Count\n"
                f"Matrix {int(matrix_meta['m'])}x{int(matrix_meta['n'])}, nnz={int(matrix_meta['nnz_a'])}, "
                f"B sparsity: {sparsity_label}"
            ),
            fontsize=14,
            fontweight="bold",
        )
        for ax, algo in zip(axes.flatten(), ["row-wise", "outer-product", "blocked", "scipy"]):
            algo_df = matrix_df[matrix_df["algorithm"] == algo]
            for k in sorted(algo_df["k"].unique()):
                series = (
                    algo_df[algo_df["k"] == k]
                    .groupby("num_procs")["mean_time_sec"]
                    .mean()
                    .sort_index()
                )
                ax.plot(series.index, series.values, marker="o", linewidth=2, label=f"k={int(k)}")
            ax.set_title(algo)
            ax.set_xlabel("Processes")
            ax.set_ylabel("Mean Runtime (seconds)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"spmm_sparse_mpi_procs_by_k_{matrix_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {plot_path}")
        plt.close()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            (
                f"{matrix_name}: Runtime vs nnz(B) by Process Count\n"
                f"Matrix {int(matrix_meta['m'])}x{int(matrix_meta['n'])}, nnz={int(matrix_meta['nnz_a'])}"
            ),
            fontsize=14,
            fontweight="bold",
        )
        for ax, algo in zip(axes.flatten(), ["row-wise", "outer-product", "blocked", "scipy"]):
            algo_df = matrix_df[matrix_df["algorithm"] == algo]
            for num_procs in sorted(algo_df["num_procs"].unique()):
                series = (
                    algo_df[algo_df["num_procs"] == num_procs]
                    .groupby("nnz_b")["mean_time_sec"]
                    .mean()
                    .sort_index()
                )
                ax.plot(
                    series.index,
                    series.values,
                    marker="o",
                    linewidth=2,
                    label=f"procs={int(num_procs)}",
                )
            ax.set_title(algo)
            ax.set_xlabel("nnz(B)")
            ax.set_ylabel("Mean Runtime (seconds)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"spmm_sparse_mpi_runtime_by_nnz_{matrix_name}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {plot_path}")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse-B SpMM with MPI on HPC, mirroring the OpenMP benchmark."
    )
    parser.add_argument("--mode", choices=["launcher", "worker", "plot"], default="launcher")
    parser.add_argument(
        "--output",
        "-o",
        default="benchmark_spmm_sparse_mpi.csv",
        help="Output CSV filename (default: benchmark_spmm_sparse_mpi.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="results_hpc_spmm_mpi",
        help="Output directory (default: results_hpc_spmm_mpi)",
    )
    parser.add_argument(
        "--cache-dir",
        default="../matrix_matrix_mult/matrices/",
        help="Matrix cache location (default: ../matrix_matrix_mult/matrices/)",
    )
    parser.add_argument(
        "--repeats",
        "-r",
        type=int,
        default=3,
        help="Repetitions per benchmark (default: 3)",
    )
    parser.add_argument(
        "--b-cols",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="Column counts for sparse B (default: 4 8 16)",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[0.10],
        help="Sparse B density levels (default: 0.10)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="MPI process counts to benchmark (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--matrices",
        nargs="+",
        default=HPC_DEFAULT_MATRICES,
        help=f"Matrix names to benchmark (default: {' '.join(HPC_DEFAULT_MATRICES)})",
    )
    parser.add_argument(
        "--mpi-launcher",
        default="mpirun",
        help="MPI launcher command to use in launcher mode (default: mpirun)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate MPI output against serial SciPy on root (expensive).",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional CSV path to plot in plot mode. Defaults to <outdir>/<output>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "plot":
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available; skipping plot generation")
            return

        csv_path = args.input_csv or os.path.join(args.outdir, args.output)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Plot input CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"No rows found in {csv_path}; skipping plot generation")
            return

        print(f"Generating MPI plots from {csv_path}...")
        plot_results(df, args.outdir)
        return

    if args.mode == "worker":
        benchmark_worker(
            cache_dir=args.cache_dir,
            matrix_names=args.matrices,
            b_cols=args.b_cols,
            sparsities=args.sparsity,
            repeats=args.repeats,
            output_csv=args.output,
            validate=args.validate,
        )
        return

    launch_worker_runs(args)


if __name__ == "__main__":
    main()
