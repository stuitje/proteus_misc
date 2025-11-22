#!/usr/bin/env python3
import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.ndimage import gaussian_filter1d
from spectres import spectres


def compute_sigma_and_new_grid(example_file, R_native, R_target, factor):
    """
    From spectrum:
    - measure Δln lambda
    - compute Gaussian sigma in pixels to degrade R_native -> R_target
    - build a new log-lambda grid with Δln lambda_new = factor * Δln lambda_old (downsampling)
    """
    with open(example_file) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(example_file, comments="#", delimiter=delim)

    lam = data[:, 0].astype(float)
    lnlam = np.log(lam)
    dln = np.median(np.diff(lnlam))

    if R_target >= R_native:
        raise ValueError(f"R_target ({R_target}) must be < R_native ({R_native})")

    # Gaussian: FWHM = 2 * sqrt(2 ln 2) * sigma
    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    sigma_ln_native = 1.0 / (R_native * fwhm_factor)
    sigma_ln_target = 1.0 / (R_target * fwhm_factor)

    # convolve with Gaussian of sigma_kernel where:
    sigma_ln_kernel2 = sigma_ln_target**2 - sigma_ln_native**2
    if sigma_ln_kernel2 <= 0:
        raise ValueError("Computed kernel sigma^2 <= 0; check R_native and R_target")

    sigma_ln_kernel = np.sqrt(sigma_ln_kernel2)
    sigma_pix = sigma_ln_kernel / dln

    # New log-lambda grid, downsampled by "factor" in Δln lambda
    dln_new = factor * dln
    ln_min = lnlam.min()
    ln_max = lnlam.max()
    n_new = int(np.floor((ln_max - ln_min) / dln_new)) + 1
    ln_new = ln_min + dln_new * np.arange(n_new)
    lam_new = np.exp(ln_new)

    return sigma_pix, dln, lam_new


def process_one_file(
    infile, outdir, sigma_pix, lam_new, R_native, R_target, factor, overwrite=False
):
    base = os.path.basename(infile)
    root, ext = os.path.splitext(base)
    out_name = f"{root}_R{int(R_target):05d}"
    outfile = os.path.join(outdir, out_name)

    if (not overwrite) and os.path.exists(outfile):
        return f"Skipping {outfile} (exists)"

    # Load data
    with open(infile) as f:
        first = f.readline()
    delim = "\t" if "\t" in first else None
    data = np.loadtxt(infile, comments="#", delimiter=delim)

    lam = data[:, 0].astype(float)
    flux = data[:, 1].astype(float)

    # 1) Degrade true resolution (Gaussian in ln lambda  -> Gaussian in pixels here)
    flux_smooth = gaussian_filter1d(flux, sigma_pix, mode="nearest")

    # 2) Flux-conserving resample onto new log-lambda grid
    lam_min = lam.min()
    lam_max = lam.max()
    mask = (lam_new >= lam_min) & (lam_new <= lam_max)
    lam_new_masked = lam_new[mask]

    # use spectres to resample
    flux_rebinned = spectres(lam_new_masked, lam, flux_smooth, fill=0.0)

    header = (
        "wavelength (nm)\tflux (erg/cm^2/s/nm)\n"
        f"Degraded from R_native={R_native:.0f} to R_target={R_target:.0f}, "
        f"Gaussian sigma_pix≈{sigma_pix:.3f}, "
        f"log-lambda step increased by factor {factor} (flux-conserving resampling)"
    )
    np.savetxt(
        outfile,
        np.column_stack([lam_new_masked, flux_rebinned]),
        fmt="%.9f\t%.6e",
        header=header,
        comments="# ",
    )
    return f"saved {outfile}"


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Degrade true spectral resolution (Gaussian in ln lambda) and "
            "then flux-conservingly resample to a new log-lambda grid "
            "(downsampling keeping constant R)."
        )
    )
    ap.add_argument(
        "--indir", default="out_compressed",
        help="Input directory with log-lambda spectra (default: out_compressed/)",
    )
    ap.add_argument(
        "--outdir", default=None,
        help="Output directory (default: out_R<Rtarget>_fac<factor>/)",
    )
    ap.add_argument(
        "--Rnative", type=float, default=10000.0,
        help="Native intrinsic resolving power (default: 10000)",
    )
    ap.add_argument(
        "--Rtarget", type=float, required=True,
        help="Target intrinsic resolving power (must be < Rnative)",
    )
    ap.add_argument(
        "--factor", type=float, default=2.0,
        help="Factor by which to increase Δln lambda (sampling step); "
             "factor=2 ≈ downsample by 2 (default: 2).",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files in outdir if they exist.",
    )
    ap.add_argument(
        "--nproc", type=int, default=None,
        help="Number of parallel processes (default: all CPUs)",
    )
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "*.txt")))
    if not files:
        print(f"No .txt files found in {args.indir}")
        return

    if args.outdir is None:
        args.outdir = f"out_R{int(args.Rtarget):05d}"
    os.makedirs(args.outdir, exist_ok=True)

    # Compute sigma_pix and the new wavelength grid once, from the first spectrum
    sigma_pix, dln_old, lam_new = compute_sigma_and_new_grid(
        files[0],
        R_native=args.Rnative,
        R_target=args.Rtarget,
        factor=args.factor,
    )
    print(
        f"Example file: {os.path.basename(files[0])}\n"
        f"  original dln ~ {dln_old:.3e}, "
        f"new dln ~ {np.median(np.diff(np.log(lam_new))):.3e}\n"
        f"  Gaussian sigma_pix ~ {sigma_pix:.3f} "
        f"(R_native={args.Rnative:.0f} -> R_target={args.Rtarget:.0f}), "
        f"sampling step x {args.factor}"
    )

    worker = partial(
        process_one_file,
        outdir=args.outdir,
        sigma_pix=sigma_pix,
        lam_new=lam_new,
        R_native=args.Rnative,
        R_target=args.Rtarget,
        factor=args.factor,
        overwrite=args.overwrite,
    )

    nproc = args.nproc or cpu_count()
    print(f"Processing {len(files)} files using {nproc} processes...\n")

    with Pool(processes=nproc) as pool:
        for msg in pool.imap_unordered(worker, files):
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
