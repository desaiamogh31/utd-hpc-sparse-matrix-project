"""
Load real sparse matrices from SuiteSparse Matrix Collection.

Download matrices from https://sparse.tamu.edu/
Popular test matrices:
  - rgg_15 (14,285 × 14,285, 60,788 non-zeros) - random geometric graph
  - rgg_20 (1,048,576 × 1,048,576) - very large
  - cage10 (11,397 × 11,397, 150,645 non-zeros) - structural problem
  - bcsstk39 (46,772 × 46,772, 2,239,490 non-zeros) - structural mechanics
"""

from __future__ import annotations
import os
import urllib.request
import io
import tarfile
import shutil
from typing import Tuple
from pathlib import Path
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix

try:
    import ssgetpy
    HAS_SSGETPY = True
except ImportError:
    HAS_SSGETPY = False


def download_matrix(group: str, name: str, cache_dir: str = "matrices") -> str:
    """
    Download a matrix from SuiteSparse Matrix Collection.
    
    Parameters:
    - group: Matrix group (e.g., "DIMACS10", "Mitsubishi")
    - name: Matrix name (e.g., "rgg_15")
    - cache_dir: Directory to cache downloaded matrices
    
    Returns:
    - Path to downloaded matrix file (.mtx)
    
    Example:
    >>> path = download_matrix("DIMACS10", "rgg_15")
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"{name}.mtx"
    filepath = os.path.join(cache_dir, filename)
    
    # Return if already cached
    if os.path.exists(filepath):
        print(f"Loaded {filename} from cache: {filepath}")
        return filepath
    
    # Try multiple URL formats
    urls = [
        # Format 1: Direct .mtx file
        f"https://sparse.tamu.edu/files/{group}/{name}/{name}.mtx",
        # Format 2: data directory
        f"https://sparse.tamu.edu/data/{group}/{name}/{name}.mtx",
        # Format 3: tar.gz archive
        f"https://sparse.tamu.edu/files/{group}/{name}/{name}.tar.gz",
    ]
    
    downloaded = False
    for url in urls:
        try:
            print(f"Trying {url}...")
            response = urllib.request.urlopen(url, timeout=30)
            data = response.read()
            
            # If it's a tar.gz, extract it
            if url.endswith('.tar.gz'):
                print("  Extracting tar.gz archive...")
                extract_dir = Path(cache_dir) / name
                extract_dir.mkdir(exist_ok=True)
                
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    tar.extractall(extract_dir)
                
                # Find the .mtx file in the extracted directory
                mtx_files = list(extract_dir.rglob("*.mtx"))
                if mtx_files:
                    source_mtx = mtx_files[0]
                    # Copy to cache location
                    import shutil
                    shutil.copy(source_mtx, filepath)
                    print(f"Extracted and cached: {filepath}")
                    downloaded = True
                    break
            else:
                # Save directly
                with open(filepath, 'wb') as f:
                    f.write(data)
                
                print(f"Downloaded and cached: {filepath}")
                downloaded = True
                break
        
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    if not downloaded and HAS_SSGETPY:
        # Fallback: try ssgetpy
        print(f"\nDirect download failed. Trying ssgetpy fallback...")
        try:
            result = ssgetpy.search(group=group, name=name, limit=1)
            if result:
                matrix = result[0]
                print(f"Found via ssgetpy: {matrix}")
                files = matrix.download(destpath=cache_dir)
                archive = Path(files[0])
                
                extract_dir = archive.parent / archive.stem
                extract_dir.mkdir(exist_ok=True)
                
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(extract_dir)
                
                mtx_files = list(extract_dir.rglob("*.mtx"))
                if mtx_files:
                    import shutil
                    shutil.copy(mtx_files[0], filepath)
                    print(f"Downloaded via ssgetpy and cached: {filepath}")
                    downloaded = True
        except Exception as e:
            print(f"  ssgetpy fallback failed: {e}")
    
    if not downloaded:
        print(f"\nError: Could not download {group}/{name}")
        print("\nTry one of these:")
        print("  1. Browse available matrices: https://sparse.tamu.edu/")
        print("  2. Use ssgetpy mode: python benchmark_suite_sparse.py ssgetpy")
        print("  3. Download manually and place in 'matrices/' folder")
        print("\nExample of correctly named matrices:")
        print("  - DIMACS10/rgg_15")
        print("  - Florida/FL_t99 (smaller)")
        raise FileNotFoundError(f"Could not download {group}/{name}")


def load_matrix_mtx(filepath: str, load_as_csr: bool = True) -> Tuple:
    """
    Load sparse matrix from Matrix Market (.mtx) file.
    
    Parameters:
    - filepath: Path to .mtx file
    - load_as_csr: Convert to CSR format if True
    
    Returns:
    - (A, info_dict): Sparse matrix and metadata
    """
    print(f"Loading matrix from {filepath}...")
    A = mmread(filepath)
    
    if load_as_csr and not isinstance(A, csr_matrix):
        A = A.tocsr()
    
    info = {
        "shape": A.shape,
        "nnz": A.nnz,
        "density": A.nnz / (A.shape[0] * A.shape[1]),
    }
    print(f"  Shape: {A.shape}, NNZ: {A.nnz}, Density: {info['density']:.2e}")
    
    return A, info


def load_matrix_ssgetpy(search_params: dict = None, cache_dir: str = "matrices") -> Tuple[csr_matrix, dict]:
    """
    Download and load a matrix from SuiteSparse using ssgetpy.
    
    Uses the SuiteSparse matrix collection API for reliable downloads.
    
    Parameters:
    - search_params: Dictionary of search parameters for ssgetpy.search()
      Examples:
        - {"nzbounds": (1000, 10000), "isspd": False, "limit": 1}
        - {"kind": "structural", "nzbounds": (1000, 10000)}
        - {"dtype": "complex"}
    - cache_dir: Directory to cache downloaded matrices
    
    Returns:
    - (A, info): CSR sparse matrix and metadata
    
    Example:
    >>> A, info = load_matrix_ssgetpy({"nzbounds": (1000, 10000), "isspd": False, "limit": 1})
    """
    if not HAS_SSGETPY:
        raise ImportError("ssgetpy not installed. Install with: pip install ssgetpy")
    
    if search_params is None:
        search_params = {"nzbounds": (1000, 10000), "isspd": False, "limit": 1}
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # 1. Search for matrix
    print(f"Searching SuiteSparse collection with params: {search_params}")
    result = ssgetpy.search(**search_params)
    
    if not result:
        raise ValueError(f"No matrices found matching: {search_params}")
    
    small_matrix = result[0]
    print(f"Found: {small_matrix}")
    print(f"  NNZ: {small_matrix.nnz}")
    
    # 2. Download archive
    print(f"Downloading {small_matrix.name}...")
    files = small_matrix.download(destpath=cache_dir)
    archive = Path(files[0])
    print(f"Downloaded: {archive}")
    
    # 3. Extract archive
    extract_dir = archive.parent / archive.stem
    extract_dir.mkdir(exist_ok=True)
    
    print(f"Extracting to {extract_dir}...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    
    # 4. Find the Matrix Market file
    mtx_files = list(extract_dir.rglob("*.mtx"))
    if not mtx_files:
        raise FileNotFoundError(f"No .mtx file found in {extract_dir}")
    
    mtx_file = mtx_files[0]
    print(f"Matrix file: {mtx_file}")
    
    # 5. Load into SciPy sparse matrix
    A = mmread(str(mtx_file)).tocsr()
    print(f"  Shape: {A.shape}, NNZ: {A.nnz}")
    
    # 6. Store a local copy
    output_filename = f"{small_matrix.name}.mtx"
    output_path = Path(cache_dir) / output_filename
    mmwrite(str(output_path), A)
    print(f"Saved to: {output_path}")
    
    info = {
        "shape": A.shape,
        "nnz": A.nnz,
        "density": A.nnz / (A.shape[0] * A.shape[1]),
        "name": small_matrix.name,
        "group": small_matrix.group,
    }
    
    return A, info


def load_suite_sparse_matrix(group: str, name: str, 
                             cache_dir: str = "matrices") -> Tuple[csr_matrix, dict]:
    """
    Download and load a matrix from SuiteSparse Matrix Collection.
    
    Parameters:
    - group: Matrix group (e.g., "DIMACS10", "Mitsubishi")
    - name: Matrix name (e.g., "rgg_15")
    - cache_dir: Directory to cache matrices
    
    Returns:
    - (A, info): CSR matrix and metadata
    
    Example:
    >>> A, info = load_suite_sparse_matrix("DIMACS10", "rgg_15")
    """
    # Download
    filepath = download_matrix(group, name, cache_dir)
    
    # Load
    A, info = load_matrix_mtx(filepath, load_as_csr=True)
    
    return A, info


# Recommended test matrices (good for benchmarking)
RECOMMENDED_MATRICES = {
    # Group: name (size, sparsity, description)
    "DIMACS10": {
        "rgg_15": (14_285, "geometric graph"),
        "road_usa": (23_947_347, "road network (very large)"),
    },
    "Mitsubishi": {
        "t520": (520, "thermal model"),
    },
    "Canary": {
        "adult": (48_842, "sparse data"),
    },
}


def print_available_matrices():
    """Print recommended matrices for benchmarking."""
    print("\nRecommended matrices for benchmarking:")
    print("=" * 60)
    for group, matrices in RECOMMENDED_MATRICES.items():
        print(f"\n{group}:")
        for name, (size, desc) in matrices.items():
            print(f"  - {name:20s} (size ~{size:,}, {desc})")
    print("\nUsage: A, info = load_suite_sparse_matrix('DIMACS10', 'rgg_15')")
    print("=" * 60)


def main():
    """Quick test: download and load a small matrix."""
    print_available_matrices()
    
    # Test download
    try:
        A, info = load_suite_sparse_matrix("DIMACS10", "rgg_15")
        print(f"\nLoaded matrix shape: {A.shape}, nnz: {A.nnz}")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()
