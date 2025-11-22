#!/usr/bin/env python3
import argparse
import numpy as np


def read_lambda(filename):
    """Read wavelength array (first column) from a text spectrum."""
    with open(filename) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(filename, comments="#", delimiter=delim)
    lam = data[:, 0]
    return lam


def analyze_grid(lam):
    """Return a dict with grid and R statistics."""
    lam = np.asarray(lam, dtype=float)
    # Sort just in case
    order = np.argsort(lam)
    lam = lam[order]

    dlam = np.diff(lam)
    lnlam = np.log(lam)
    dln = np.diff(lnlam)

    # Sampling resolving power from linear spacing
    lam_mid = 0.5 * (lam[:-1] + lam[1:])
    R_samp = lam_mid / dlam

    stats = {}

    stats["lam_min"] = lam.min()
    stats["lam_max"] = lam.max()
    stats["n_points"] = lam.size

    stats["dlam_mean"] = dlam.mean()
    stats["dlam_frac_std"] = dlam.std() / dlam.mean()

    stats["dln_mean"] = dln.mean()
    stats["dln_frac_std"] = dln.std() / dln.mean()

    stats["R_median"] = np.median(R_samp)
    stats["R_min"] = R_samp.min()
    stats["R_max"] = R_samp.max()
    stats["R_frac_span"] = (stats["R_max"] - stats["R_min"]) / stats["R_median"]

    # Classify grid type
    # "Constant" here means "fractional scatter < 1%".
    frac_linear = stats["dlam_frac_std"]
    frac_log = stats["dln_frac_std"]

    if frac_linear < 0.01 and frac_linear < frac_log:
        grid_type = "approximately linear in λ (constant Δλ)"
        R_constant = False
    elif frac_log < 0.01 and frac_log < frac_linear:
        grid_type = "approximately logarithmic in λ (constant Δlnλ → constant R sampling)"
        R_constant = True
    else:
        grid_type = "irregular / mixed (neither Δλ nor Δlnλ is very constant)"
        R_constant = False

    stats["grid_type"] = grid_type
    stats["R_constant"] = R_constant

    # If log grid, estimate constant R from dln
    if R_constant:
        stats["R_from_dln"] = 1.0 / stats["dln_mean"]
    else:
        stats["R_from_dln"] = None

    return stats


def print_report(filename, stats):
    print(f"\n=== {filename} ===")
    print(f"N points: {stats['n_points']}")
    print(f"λ range: {stats['lam_min']:.3f} – {stats['lam_max']:.3f}")

    print(f"\nGrid statistics:")
    print(f"  ⟨Δλ⟩       = {stats['dlam_mean']:.6e}")
    print(f"  frac std(Δλ)  = {stats['dlam_frac_std']:.3e}")
    print(f"  ⟨Δln λ⟩    = {stats['dln_mean']:.6e}")
    print(f"  frac std(Δln λ) = {stats['dln_frac_std']:.3e}")

    print(f"\nSampling R (R_samp ≈ λ / Δλ):")
    print(f"  median R   = {stats['R_median']:.1f}")
    print(f"  min R      = {stats['R_min']:.1f}")
    print(f"  max R      = {stats['R_max']:.1f}")
    print(f"  (max - min)/median = {stats['R_frac_span']:.3e}")

    print(f"\nGrid classification:")
    print(f"  {stats['grid_type']}")
    if stats["R_constant"]:
        print(f"  → Estimated constant R ≈ 1/⟨Δln λ⟩ ≈ {stats['R_from_dln']:.1f}")
    else:
        print(f"  → Sampling R is NOT constant across the range.")


def main():
    ap = argparse.ArgumentParser(
        description="Inspect wavelength grid of spectral files and estimate sampling R and whether it is constant."
    )
    ap.add_argument("files", nargs="+", help="Spectral text files (wavelength in first column).")
    args = ap.parse_args()

    for fname in args.files:
        lam = read_lambda(fname)
        stats = analyze_grid(lam)
        print_report(fname, stats)


if __name__ == "__main__":
    main()
