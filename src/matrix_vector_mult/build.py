#!/usr/bin/env python3
"""
Build script for OpenMP-accelerated SpMV.

Compiles spmv_openmp.cpp to a shared library (spmv_openmp.so)
using gcc/clang with OpenMP support.
"""

import subprocess
import os
import sys
import platform


def detect_compiler():
    """Detect available compiler (gcc, mpicxx, or clang with OpenMP)."""
    compilers = []
    
    # Try gcc-11, gcc-12, gcc first (GNU compilers have native -fopenmp)
    for gcc_variant in ["gcc-12", "gcc-11", "gcc-10", "gcc"]:
        try:
            result = subprocess.run([gcc_variant, "--version"], capture_output=True, text=True)
            if result.returncode == 0 and "GCC" in result.stdout:
                compilers.append((gcc_variant, "gcc"))
        except FileNotFoundError:
            pass
    
    # Try mpicxx (MPI compiler, usually has OpenMP support)
    try:
        result = subprocess.run(["mpicxx", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            compilers.append(("mpicxx", "mpicxx"))
    except FileNotFoundError:
        pass
    
    # Try clang with libomp (installed via Homebrew on macOS)
    try:
        result = subprocess.run(["clang", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            # Check if libomp is available
            try:
                subprocess.run(["clang", "-fopenmp", "-c", "-x", "c", "-"], 
                             input="int main() {}", capture_output=True, text=True)
                compilers.append(("clang", "clang"))
            except:
                pass
    except FileNotFoundError:
        pass
    
    return compilers


def build():
    """Compile spmv_openmp.cpp to shared library."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_file = os.path.join(script_dir, "spmv_openmp.cpp")
    so_file = os.path.join(script_dir, "spmv_openmp.so")
    
    if not os.path.exists(cpp_file):
        print(f"Error: {cpp_file} not found")
        return False
    
    print("=" * 70)
    print("Building OpenMP SpMV shared library")
    print("=" * 70)
    print(f"Input:  {cpp_file}")
    print(f"Output: {so_file}")
    print("-" * 70)
    
    # Detect available compilers
    compilers = detect_compiler()
    
    if not compilers:
        print("✗ Error: No suitable compiler found with OpenMP support")
        print("\nAvailable options:")
        print("1. On macOS with MPI installed: mpicxx is recommended (you likely have it)")
        print("   Run: which mpicxx")
        print("\n2. On macOS without Homebrew:")
        print("   Install gcc+OpenMP: brew install gcc libomp")
        print("\n3. On Linux:")
        print("   Ubuntu/Debian: sudo apt-get install build-essential libomp-dev")
        print("   RedHat/CentOS: sudo yum install gcc gcc-c++ openmp-devel")
        return False
    
    # Try each compiler
    for compiler, compiler_type in compilers:
        print(f"Trying {compiler}...")
        
        # Common compiler flags
        flags = ["-O3", "-fPIC", "-shared"]
        
        # macOS specific: need to check if we're using clang and add libomp
        if platform.system() == "Darwin" and compiler_type == "clang":
            # Try with -fopenmp (requires libomp)
            flags.extend(["-fopenmp"])
            # Try to link libomp from Homebrew
            try:
                result = subprocess.run(["brew", "--prefix", "libomp"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    libomp_path = result.stdout.strip()
                    flags.extend([f"-I{libomp_path}/include", f"-L{libomp_path}/lib"])
            except:
                pass
        else:
            # GNU gcc or mpicxx: use standard -fopenmp
            flags.append("-fopenmp")
        
        # Add optimization flag for native CPU (not for mpicxx, it's not always supported)
        if compiler_type != "mpicxx":
            flags.insert(1, "-march=native")
        else:
            flags.insert(1, "-march=native")  # Try anyway
        
        cmd = [compiler] + flags + ["-o", so_file, cpp_file]
        
        print(f"Command: {' '.join(cmd)}")
        print("-" * 70)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(so_file):
                print("-" * 70)
                print(f"✓ Build successful with {compiler}")
                print(f"  Output: {so_file}")
                print(f"  File size: {os.path.getsize(so_file) / 1024:.1f} KB")
                print("=" * 70)
                return True
            else:
                print(f"✗ {compiler} failed:")
                print(result.stderr)
                print("-" * 70)
                
        except subprocess.TimeoutExpired:
            print(f"✗ {compiler} timed out")
            print("-" * 70)
        except Exception as e:
            print(f"✗ {compiler} error: {e}")
            print("-" * 70)
    
    print("✗ Build failed: all compilers failed")
    print("\nTroubleshooting:")
    print("1. Ensure gcc or clang is installed")
    print("2. On macOS: brew install gcc libomp")
    print("3. Check that OpenMP headers are available")
    return False


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
