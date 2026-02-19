#!/usr/bin/env python3
"""
src/eigen.py

2D finite-difference Hamiltonian on an N x N grid with optional Dirichlet boundaries.

- Eigenpairs are computed for the homogeneous Dirichlet problem on the interior unknowns phi:
      H_int phi = E phi,   with phi = 0 on the boundary.
- If you additionally want a fixed nonzero boundary value psi|boundary = f(x,y)=a x + b y,
  we reconstruct a field psi = phi + f for output/visualization. The spectrum is still the
  homogeneous Dirichlet spectrum (physically meaningful).
"""

import argparse
import os
import numpy as np
from scipy.linalg import eigh


def build_potential(
    N: int,
    potential: str,
    harmonic_k: float,
    quartic_k: float,
    quartic_anis: float,
) -> np.ndarray:
    """
    Return V on the full N x N grid.

    Coordinates:
      - For harmonic/quartic we use centered coordinates like earlier:
            x = (i - N/2)*dx, y = (j - N/2)*dx, with dx = 1/N.
      - For 'well' we return V=0 everywhere.

    Potentials implemented:
      well:    V=0
      harmonic: V = harmonic_k * (x^2 + y^2)
      quartic:  V = quartic_k * (x^4 + quartic_anis * y^4)
    """
    dx = 1.0 / float(N)
    V = np.zeros((N, N), dtype=np.float64)

    if potential == "well":
        return V

    xs = (np.arange(N) - N / 2.0) * dx
    ys = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    if potential == "harmonic":
        return harmonic_k * (X * X + Y * Y)

    if potential == "quartic":
        if quartic_anis <= 0:
            raise ValueError(f"quartic_anis must be > 0, got {quartic_anis}.")
        return quartic_k * (X**4 + quartic_anis * (Y**4))

    raise ValueError(f"Unknown potential={potential!r}.")


def boundary_function(N: int, a: float, b: float) -> np.ndarray:
    """
    f(x,y)=a x + b y on the full grid, with x,y in [0,1] using dx=1/N convention.
    """
    dx = 1.0 / float(N)
    xs = np.arange(N) * dx
    ys = np.arange(N) * dx
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return a * X + b * Y


def build_dirichlet_interior_hamiltonian(N: int, Vfull: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Build H on interior unknowns only: (i,j) = 1..N-2.
    This enforces homogeneous Dirichlet on the boundary for the eigenproblem.
    """
    if N < 3:
        raise ValueError("Dirichlet interior requires N >= 3.")

    inv_dx2 = float(N * N)

    Nint = N - 2
    dim = Nint * Nint
    H = np.zeros((dim, dim), dtype=np.float64)

    def idx(ii: int, jj: int) -> int:
        return ii * Nint + jj  # ii,jj in 0..Nint-1

    for ii in range(Nint):
        for jj in range(Nint):
            i = ii + 1
            j = jj + 1
            row = idx(ii, jj)

            H[row, row] = 4.0 * inv_dx2 + Vfull[i, j]

            if ii > 0:
                H[row, idx(ii - 1, jj)] = -inv_dx2
            if ii < Nint - 1:
                H[row, idx(ii + 1, jj)] = -inv_dx2
            if jj > 0:
                H[row, idx(ii, jj - 1)] = -inv_dx2
            if jj < Nint - 1:
                H[row, idx(ii, jj + 1)] = -inv_dx2

    return H, Nint


def solve_eigen(
    N: int,
    potential: str,
    neigs: int,
    bc: str,
    harmonic_k: float,
    quartic_k: float,
    quartic_anis: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if N <= 0:
        raise ValueError("N must be positive.")
    if potential not in {"well", "harmonic", "quartic"}:
        raise ValueError("potential must be 'well', 'harmonic', or 'quartic'.")
    if bc not in {"none", "dirichlet"}:
        raise ValueError("bc must be 'none' or 'dirichlet'.")
    if neigs <= 0:
        raise ValueError("neigs must be positive.")

    Vfull = build_potential(
        N,
        potential,
        harmonic_k=harmonic_k,
        quartic_k=quartic_k,
        quartic_anis=quartic_anis,
    )

    if bc == "none":
        # Full-grid unknowns.
        inv_dx2 = float(N * N)
        dim = N * N
        H = np.zeros((dim, dim), dtype=np.float64)

        def idx(i: int, j: int) -> int:
            return i * N + j

        for i in range(N):
            for j in range(N):
                row = idx(i, j)
                H[row, row] = 4.0 * inv_dx2 + Vfull[i, j]
                if i > 0:
                    H[row, idx(i - 1, j)] = -inv_dx2
                if i < N - 1:
                    H[row, idx(i + 1, j)] = -inv_dx2
                if j > 0:
                    H[row, idx(i, j - 1)] = -inv_dx2
                if j < N - 1:
                    H[row, idx(i, j + 1)] = -inv_dx2

        vals, vecs = eigh(H)
        order = np.argsort(vals)
        vals = vals[order][:neigs]
        vecs = vecs[:, order][:, :neigs]
        return vals, vecs, N  # Nint = N

    # bc == "dirichlet"
    H, Nint = build_dirichlet_interior_hamiltonian(N, Vfull)
    dim = Nint * Nint
    if neigs > dim:
        raise ValueError(f"neigs={neigs} too large for interior dim={dim}.")

    vals, vecs = eigh(H)
    order = np.argsort(vals)
    vals = vals[order][:neigs]
    vecs = vecs[:, order][:, :neigs]
    return vals, vecs, Nint


def reconstruct_full_field_from_interior(phi_vec: np.ndarray, N: int, Nint: int) -> np.ndarray:
    """
    Put interior vector (length Nint^2) into an N x N array with zeros on boundary.
    """
    phi_full = np.zeros((N, N), dtype=np.float64)
    phi_int = phi_vec.reshape(Nint, Nint)
    phi_full[1:-1, 1:-1] = phi_int
    return phi_full


def str2bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=20)
    p.add_argument("--potential", choices=["well", "harmonic", "quartic"], default="well")
    p.add_argument("--neigs", type=int, default=5)
    p.add_argument("--bc", choices=["none", "dirichlet"], default="dirichlet")
    p.add_argument("--GS", type=str, default="False", help="If True, save ground-state |psi|^2 on NxN grid.")
    p.add_argument("--a", type=float, default=0.0, help="Boundary f(x,y)=a x + b y")
    p.add_argument("--b", type=float, default=0.0, help="Boundary f(x,y)=a x + b y")

    # Potential parameters (kept explicit so you can vary them in your study)
    p.add_argument("--harmonic-k", type=float, default=100.0, help="Strength for harmonic: V=k(x^2+y^2).")
    p.add_argument("--quartic-k", type=float, default=100.0, help="Strength for quartic: V=k(x^4+anis*y^4).")
    p.add_argument("--quartic-anis", type=float, default=1.0, help="Anisotropy factor (>0) multiplying y^4 term.")

    return p.parse_args()


def main():
    args = parse_args()
    gs_flag = str2bool(args.GS)

    os.makedirs("results", exist_ok=True)

    vals, vecs, Nint = solve_eigen(
        N=args.N,
        potential=args.potential,
        neigs=args.neigs,
        bc=args.bc,
        harmonic_k=args.harmonic_k,
        quartic_k=args.quartic_k,
        quartic_anis=args.quartic_anis,
    )

    # Save eigenvalues
    eig_path = f"results/eigs_N{args.N}_{args.potential}_bc{args.bc}.txt"
    np.savetxt(eig_path, vals)

    print(f"N={args.N}, potential={args.potential}, bc={args.bc}, neigs={args.neigs}")
    print(f"Saved eigenvalues -> {eig_path}")
    for k, v in enumerate(vals, start=1):
        print(f"{k:2d}: {v:.16e}")

    if gs_flag:
        if args.bc != "dirichlet":
            raise ValueError("Ground-state reconstruction with boundary f(x,y) is implemented for bc='dirichlet' only.")

        # Ground eigenvector is column 0 in vecs
        phi0 = vecs[:, 0]
        phi_full = reconstruct_full_field_from_interior(phi0, args.N, Nint)

        # Enforce requested Dirichlet boundary on the SAVED field via psi = phi + f
        f = boundary_function(args.N, args.a, args.b)
        psi_full = phi_full + f

        psi2 = psi_full * psi_full  # real case
        gs_path = (
            f"results/groundstate_prob_N{args.N}_{args.potential}_bc{args.bc}"
            f"_a{args.a}_b{args.b}.txt"
        )
        np.savetxt(gs_path, psi2)

        print(f"Saved ground-state probability density |psi|^2 -> {gs_path}")
        print(f"(Shape {psi2.shape}, boundary satisfies psi=f(x,y)=a x + b y in the saved field.)")


if __name__ == "__main__":
    main()