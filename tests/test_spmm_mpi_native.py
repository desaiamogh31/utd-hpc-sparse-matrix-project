"""
Smoke tests for the native C++ MPI sparse-B SpMM benchmark.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CPP_FILE = REPO_ROOT / "src" / "matrix_mult_mpi" / "spmm_sparse_mpi.cpp"


@pytest.mark.skipif(shutil.which("mpicxx") is None, reason="mpicxx not available")
def test_native_mpi_benchmark_smoke(tmp_path):
    exe_path = tmp_path / "spmm_sparse_mpi"
    matrix_path = tmp_path / "toy.mtx"
    output_dir = tmp_path / "results"
    output_csv = output_dir / "out.csv"

    matrix_path.write_text(
        "\n".join(
            [
                "%%MatrixMarket matrix coordinate real general",
                "% toy matrix",
                "3 3 4",
                "1 1 1.0",
                "1 3 2.0",
                "2 2 3.0",
                "3 1 4.0",
            ]
        )
        + "\n"
    )

    try:
        subprocess.run(
            ["mpicxx", "-O3", "-std=c++17", "-o", str(exe_path), str(CPP_FILE)],
            check=True,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"mpicxx is present but not usable on this machine: {exc}")

    subprocess.run(
        [
            str(exe_path),
            "--cache-dir",
            str(tmp_path),
            "--matrices",
            "toy",
            "--b-cols",
            "2",
            "--sparsity",
            "0.5",
            "--repeats",
            "1",
            "--outdir",
            str(output_dir),
            "--output",
            "out.csv",
            "--validate",
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    df = pd.read_csv(output_csv)
    assert set(df["algorithm"]) == {"row-wise", "outer-product", "blocked", "scipy"}
    assert set(df["num_procs"]) == {1}
    assert set(df["validation_ok"]) == {True}
