#!/usr/bin/env python3
"""
src/eigen.py

Build a discretized 2D Hamiltonian on an N x N grid using a 5-point finite-difference stencil,
then compute eigenvalues/eigenvectors.

Optional:
  --GS   Save ground-state probability density |psi|^2
"""

import argparse
import os
import numpy as np
from scipy.linalg import eigh


def build_2d_hamiltonian(N: int = 20, potential: str = "well") -> np.ndarray:

    dx = 1.0 / float(N)
    inv_dx2 = float(N * N)

    H = np.zeros((N * N, N * N), dtype=np.float64)

    def idx(i: int, j: int) -> int:
        return i * N + j

    def V(i: int, j: int) -> float:
        if potential == "well":
            return 0.0
        elif potential == "harmonic":
            x = (i - N / 2.0) * dx
            y = (j - N / 2.0) * dx
            return 100.0 * (x * x + y * y)
        else:
            return 0.0

    for i in range(N):
        for j in range(N):
            row = idx(i, j)

            H[row, row] = 4.0 * inv_dx2 + V(i, j)

            if i > 0:
                H[row, idx(i - 1, j)] = -inv_dx2
            if i < N - 1:
                H[row, idx(i + 1, j)] = -inv_dx2
            if j > 0:
                H[row, idx(i, j - 1)] = -inv_dx2
            if j < N - 1:
                H[row, idx(i, j + 1)] = -inv_dx2

    return H


def solve_eigen(N: int = 20, potential: str = "well"):

    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be positive integer")

    H = build_2d_hamiltonian(N, potential)

    vals, vecs = eigh(H)

    idx_sorted = np.argsort(vals)
    vals = vals[idx_sorted]
    vecs = vecs[:, idx_sorted]

    return vals, vecs


def parse_args():
    p = argparse.ArgumentParser(
        description="2D finite-difference Hamiltonian solver"
    )
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--potential", type=str, default="well",
                   choices=["well", "harmonic"])
    p.add_argument("--neigs", type=int, default=5)
    p.add_argument("--GS", action="store_true",
                   help="If set, save ground-state probability density")
    return p.parse_args()


def main():

    args = parse_args()

    os.makedirs("results", exist_ok=True)

    vals, vecs = solve_eigen(N=args.N, potential=args.potential)

    # Save eigenvalues
    vals_to_save = vals[:args.neigs]
    np.savetxt(f"results/eigs_N{args.N}_{args.potential}.txt", vals_to_save)

    print(f"N={args.N}, potential={args.potential}")
    print("Lowest eigenvalues:")
    for k, v in enumerate(vals_to_save, start=1):
        print(f"{k:2d}: {v:.12g}")

    # ----- OPTIONAL GROUND STATE -----
    if args.GS:

        ground_vec = vecs[:, 0]            # lowest eigenvector
        psi = ground_vec.reshape((args.N, args.N))
        prob_density = np.abs(psi) ** 2

        np.savetxt(
            f"results/groundstate_N{args.N}_{args.potential}.txt",
            prob_density
        )

        print("Ground-state probability density saved.")


if __name__ == "__main__":
    main()