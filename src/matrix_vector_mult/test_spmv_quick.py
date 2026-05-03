"""
Quick test of spmv_python.py to verify implementation.
"""
import numpy as np
from scipy.sparse import random as sparse_random
from spmv_python import spmv, validate_spmv

# Create a small test matrix and vector
print("Creating test matrix (n=100, nnz_ratio=5.0)...")
n = 100
nnz_ratio = 5.0
nnz = int(n * nnz_ratio)
density = nnz / (n * n)

A = sparse_random(n, n, density=density, format='coo', random_state=42)
x = np.random.randn(n)

print(f"Matrix: {n}×{n}, NNZ={A.nnz}")

# Test all 4 formats
formats = ["coo", "csr", "csc", "lil"]
print("\nTesting formats...")
print("-" * 60)

for fmt in formats:
    print(f"  {fmt.upper():>5}: ", end="", flush=True)
    try:
        y = spmv(A, x, fmt)
        is_valid = validate_spmv(A, x, y)
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status:>20}")
    except Exception as e:
        print(f"✗ ERROR: {e}")

print("-" * 60)
print("✓ spmv_python.py is working!")
