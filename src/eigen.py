#!/usr/bin/env python3
"""
src/eigen.py

Build a discretized 2D Hamiltonian on an N x N grid using a 5-point finite-difference stencil,
then compute eigenvalues/eigenvectors.

Usage examples:
  python src/eigen.py --N 10 --potential well --neigs 5
  python src/eigen.py --N 20 --potential harmonic --neigs 8
"""

import argparse
import numpy as np
from scipy.linalg import eigh


def build_2d_hamiltonian(N: int = 20, potential: str = "well") -> np.ndarray:
    """
    Build a discretized 2D Hamiltonian on an N x N grid.

    This follows the structure of your reference code:
      - grid spacing dx = 1/N
      - inv_dx2 = 1/dx^2 = N^2
      - 5-point stencil coupling each site to its 4 neighbors

    Parameters
    ----------
    N : int
        Number of grid points per dimension (total unknowns = N^2).
    potential : str
        Potential type: 'well' or 'harmonic'.

    Returns
    -------
    H : ndarray, shape (N^2, N^2)
        Hamiltonian matrix.
    """
    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)

    H = np.zeros((N * N, N * N), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * N + j

    def V(i: int, j: int) -> float:
        if potential == "well":
            # "Well" example: V=0 everywhere (boundary conditions not enforced here).
            return 0.0
        elif potential == "harmonic":
            # Harmonic trap centered at grid midpoint
            x = (i - N / 2.0) * dx
            y = (j - N / 2.0) * dx
            return 4.0 * (x * x + y * y)
        else:
            # Should not happen if inputs are validated, but keep safe fallback.
            return 0.0

    # Fill H using the same stencil pattern as your reference code.
    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            # Diagonal term: stencil center + potential
            H[row, row] = 4.0 * inv_dx2 + V(i, j)

            # Neighbor couplings
            if i > 0:
                H[row, idx(i - 1, j)] = -inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = -inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = -inv_dx2

    return H


def solve_eigen(N: int = 20, potential: str = "well", n_eigs: int | None = None):
    """
    Build a 2D Hamiltonian and solve for eigenpairs.

    Parameters
    ----------
    N : int
        Grid points per dimension.
    potential : str
        Potential type: 'well' or 'harmonic'.
    n_eigs : int or None
        If provided, return only the lowest n_eigs eigenvalues/eigenvectors.

    Returns
    -------
    vals : ndarray
        Sorted eigenvalues (ascending). If n_eigs is set, only the first n_eigs.
    vecs : ndarray
        Corresponding eigenvectors as columns, sorted consistently with vals.
    """
    # Sanity checks (requested)
    if not isinstance(N, int) or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N!r}.")
    if potential not in {"well", "harmonic"}:
        raise ValueError(f"potential must be one of {{'well','harmonic'}}, got {potential!r}.")
    if n_eigs is not None:
        if not isinstance(n_eigs, int) or n_eigs <= 0:
            raise ValueError(f"n_eigs must be a positive integer or None, got {n_eigs!r}.")
        if n_eigs > N * N:
            raise ValueError(f"n_eigs must satisfy n_eigs <= N^2 = {N*N}, got {n_eigs}.")

    H = build_2d_hamiltonian(N, potential)

    # Full dense diagonalization (OK for small N)
    vals, vecs = eigh(H)

    # Sort explicitly (safe)
    idx_sorted = np.argsort(vals)
    vals = vals[idx_sorted]
    vecs = vecs[:, idx_sorted]

    if n_eigs is None:
        return vals, vecs
    return vals[:n_eigs], vecs[:, :n_eigs]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a discretized 2D Hamiltonian on an N x N grid and compute eigenvalues."
    )
    p.add_argument("--N", type=int, default=20, help="Grid points per dimension (total = N^2).")
    p.add_argument(
        "--potential",
        type=str,
        default="well",
        choices=["well", "harmonic"],
        help="Potential type.",
    )
    p.add_argument(
        "--neigs",
        type=int,
        default=5,
        help="Number of lowest eigenvalues to print (must be <= N^2).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    vals, _ = solve_eigen(N=args.N, potential=args.potential, n_eigs=args.neigs)

    print(f"N={args.N}, potential={args.potential}, neigs={args.neigs}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals, start=1):
        print(f"  {k:2d}: {v:.12g}")


if __name__ == "__main__":
    main()