#!/usr/bin/env python3
import argparse
import os
import glob

import numpy as np
import astropy.units as u
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler


def process_file(infile, outdir, factor):
    # Load, header starts with '#')
    with open(infile) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.genfromtxt(infile, comments="#", delimiter=delim)

    lam_nm = data[:, 0]
    flux = data[:, 1] * (u.erg / (u.cm**2 * u.s * u.nm))
    spec = Spectrum1D(spectral_axis=lam_nm * u.nm, flux=flux)

    # Build log-lambda grid (uniform in ln lambda)
    lnlam = np.log(lam_nm)
    dln = np.median(np.diff(lnlam))
    step = factor * dln
    ln_min, ln_max = lnlam.min(), lnlam.max()
    ln_new = ln_min + step * np.arange(int((ln_max - ln_min) / step) + 1)
    lam_new = np.exp(ln_new) * u.nm

    # Flux-conserving resample in lambda, truncate edges
    resampler = FluxConservingResampler(extrapolation_treatment="truncate")
    spec_ds = resampler(spec, lam_new)

    # output path
    base = os.path.basename(infile)
    root, ext = os.path.splitext(base)
    outfile = os.path.join(outdir, f"{root}_compressed{factor}{ext}")

    np.savetxt(outfile, np.column_stack([ spec_ds.spectral_axis.to_value(u.nm), spec_ds.flux.to_value(u.erg / (u.cm**2 * u.s * u.nm)),]),
        fmt="%.9f\t%.6e", header="wavelength (nm)\tflux (erg/cm^2/s/nm)", comments="# ")
    print("saved:", outfile)

def main():
    p = argparse.ArgumentParser(
        description="Flux-conserving resample onto a log-wavelength grid for all .txt files in a directory."
    )
    p.add_argument(
        "--indir", default="out",
        help="Input directory containing .txt spectra (default: out/)"
    )
    p.add_argument(
        "--outdir", default="out_compressed",
        help="Output directory for resampled spectra (default: out_compressed/)"
    )
    p.add_argument(
        "--factor", type=int, default=5,
        help="Multiplier for the native ln(lambda) step (default: 5)"
    )
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.indir, "*.txt")))
    if not files:
        print(f"No .txt files found in {args.indir}")
        return

    for infile in files:
        try:
            process_file(infile, args.outdir, args.factor)
        except Exception as e:
            print(f"Error processing {infile}: {e}")


if __name__ == "__main__":
    main()
