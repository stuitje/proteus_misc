#!/usr/bin/env python3
import argparse
import numpy as np


def load_spectrum(filename):
    """Load wavelength and flux columns from a text spectrum file."""
    with open(filename) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(filename, comments="#", delimiter=delim)

    lam = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)
    return lam, flux


def flux_integral(lam, flux, lam_min, lam_max):
    """
    Compute the flux integral between lam_min and lam_max
    using the trapezoidal rule on the native grid.
    """
    lam = np.asarray(lam, float)
    flux = np.asarray(flux, float)

    # Restrict to the requested wavelength range
    mask = (lam >= lam_min) & (lam <= lam_max)
    lam_sel = lam[mask]
    flux_sel = flux[mask]

    if lam_sel.size < 2:
        raise ValueError(
            f"Not enough points in range [{lam_min}, {lam_max}] for integration "
            f"(only {lam_sel.size} points found)."
        )

    return np.trapz(flux_sel, lam_sel)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test flux conservation between two spectra by comparing "
            "the integrated flux between two wavelengths."
        )
    )
    parser.add_argument("spec1", help="First spectrum file (wavelength, flux).")
    parser.add_argument("spec2", help="Second spectrum file (wavelength, flux).")
    parser.add_argument(
        "--lam-min", type=float, required=True,
        help="Lower wavelength bound of integration (same units as files, e.g. nm).",
    )
    parser.add_argument(
        "--lam-max", type=float, required=True,
        help="Upper wavelength bound of integration (same units as files, e.g. nm).",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3,
        help="Relative tolerance for flux agreement (default: 1e-3).",
    )
    args = parser.parse_args()

    lam1, flux1 = load_spectrum(args.spec1)
    lam2, flux2 = load_spectrum(args.spec2)

    F1 = flux_integral(lam1, flux1, args.lam_min, args.lam_max)
    F2 = flux_integral(lam2, flux2, args.lam_min, args.lam_max)

    rel_diff = abs(F1 - F2) / max(abs(F1), abs(F2))

    print(f"File 1: {args.spec1}")
    print(f"  Integral[{args.lam_min}, {args.lam_max}] = {F1:.6e}")
    print(f"File 2: {args.spec2}")
    print(f"  Integral[{args.lam_min}, {args.lam_max}] = {F2:.6e}")
    print(f"Relative difference = {rel_diff:.3e} (rtol = {args.rtol:.3e})")

    if rel_diff <= args.rtol:
        print("SUCCESS: Flux integrals agree within tolerance.")
    else:
        raise AssertionError(
            f"FAILURE: Flux integrals differ more than rtol={args.rtol:.3e}: rel_diff={rel_diff:.3e}"
        )


if __name__ == "__main__":
    main()
