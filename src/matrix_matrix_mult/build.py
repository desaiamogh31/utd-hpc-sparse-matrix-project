#!/usr/bin/env python3
"""
Build script for OpenMP-accelerated sparse-B SpMM.

Compiles spmm_openmp.cpp to a shared library (spmm_openmp.so) for local
laptop testing.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys


def detect_compilers() -> list[tuple[str, str]]:
    """Return candidate compilers with likely OpenMP support."""
    compilers: list[tuple[str, str]] = []

    for gcc_variant in ["g++-13", "g++-12", "g++-11", "g++-10", "g++"]:
        try:
            result = subprocess.run(
                [gcc_variant, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0 and "GCC" in result.stdout:
                compilers.append((gcc_variant, "gcc"))
        except FileNotFoundError:
            pass

    try:
        result = subprocess.run(["mpicxx", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            compilers.append(("mpicxx", "mpicxx"))
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(["clang++", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            compilers.append(("clang++", "clang"))
    except FileNotFoundError:
        pass

    return compilers


def build() -> bool:
    """Compile the sparse-B SpMM OpenMP shared library."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_file = os.path.join(script_dir, "spmm_openmp.cpp")
    so_file = os.path.join(script_dir, "spmm_openmp.so")

    if not os.path.exists(cpp_file):
        print(f"Error: {cpp_file} not found")
        return False

    print("=" * 70)
    print("Building OpenMP sparse-B SpMM shared library")
    print("=" * 70)
    print(f"Input:  {cpp_file}")
    print(f"Output: {so_file}")
    print("-" * 70)

    compilers = detect_compilers()
    if not compilers:
        print("✗ Error: No suitable compiler found with OpenMP support")
        print("Try installing GCC or clang/libomp first.")
        return False

    for compiler, compiler_type in compilers:
        flags = ["-O3", "-march=native", "-fPIC", "-shared"]

        if platform.system() == "Darwin" and compiler_type == "clang":
            flags.append("-Xpreprocessor")
            flags.append("-fopenmp")
            libomp_candidates = []
            try:
                result = subprocess.run(
                    ["brew", "--prefix", "libomp"], capture_output=True, text=True
                )
                if result.returncode == 0:
                    libomp_candidates.append(result.stdout.strip())
            except FileNotFoundError:
                pass

            libomp_candidates.extend(
                ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]
            )

            for libomp_path in libomp_candidates:
                if os.path.exists(os.path.join(libomp_path, "include", "omp.h")):
                    flags.extend(
                        [
                            f"-I{libomp_path}/include",
                            f"-L{libomp_path}/lib",
                            "-lomp",
                        ]
                    )
                    break
        else:
            flags.append("-fopenmp")

        cmd = [compiler] + flags + ["-o", so_file, cpp_file]

        print(f"Trying {compiler}...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 70)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            print(f"✗ {compiler} timed out")
            print("-" * 70)
            continue
        except Exception as exc:
            print(f"✗ {compiler} error: {exc}")
            print("-" * 70)
            continue

        if result.returncode == 0 and os.path.exists(so_file):
            print(f"✓ Build successful with {compiler}")
            print(f"  File size: {os.path.getsize(so_file) / 1024:.1f} KB")
            print("=" * 70)
            return True

        print(f"✗ {compiler} failed:")
        if result.stderr:
            print(result.stderr)
        if result.stdout:
            print(result.stdout)
        print("-" * 70)

    print("✗ Build failed: all compilers failed")
    print("Troubleshooting:")
    print("1. Install gcc or clang/libomp")
    print("2. On macOS: brew install gcc libomp")
    print("3. Re-run: python build.py")
    return False


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
