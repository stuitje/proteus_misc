#!/usr/bin/env python3
import argparse
import os
import glob
import re

import numpy as np
import astropy.units as u
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

# PHOENIX input name patterns in out/
# Examples:
#   lte06000-4.00-0.0.Alpha=-0.20.PHOENIX-ACES-AGSS-COND-2011-MidRes.txt
#   lte06000-5.50+1.0.PHOENIX-ACES-AGSS-COND-2011-MidRes.txt

PAT_NO_ALPHA = re.compile(
    r"^lte(\d+)"                       # Teff, e.g. 06000
    r"([+-]\d+\.\d+)"                  # logg, e.g. 5.50
    r"([+-]\d+\.\d+)"                  # [Fe/H], e.g. -0.0, +1.0
    r"\.PHOENIX-ACES-AGSS-COND-2011-MidRes\.txt$"
)

PAT_ALPHA = re.compile(
    r"^lte(\d+)"                       # Teff
    r"([+-]\d+\.\d+)"                  # logg
    r"([+-]\d+\.\d+)"                  # [Fe/H]
    r"\.Alpha=([+-]\d+\.\d+)"          # [alpha/Fe]
    r"\.PHOENIX-ACES-AGSS-COND-2011-MidRes\.txt$"
)


def build_outname_phoenix(base):
    """
    Given a PHOENIX filename (basename), build the LTE_T..._phoenixMidRes_compressed.txt name.
    Returns None if it doesn't match the expected PHOENIX patterns.
    """
    m = PAT_ALPHA.match(base)
    if m:
        teff, logg, metal, alpha = m.groups()
    else:
        m = PAT_NO_ALPHA.match(base)
        if not m:
            return None
        teff, logg, metal = m.groups()
        alpha = "+0.0"  # default [alpha/Fe] when missing

    # numeric values for consistent formatting
    teff_val = int(teff)
    logg_val = abs(float(logg))      # logg always positive
    metal_val = float(metal)
    alpha_val = float(alpha)

    # Desired format:
    #   LTE_T06000_logg4.00_FeH-0.0_alpha-0.2_phoenixMidRes_compressed.txt
    teff_str = f"{teff_val:05d}"         # 6000 -> "06000"
    logg_str = f"{logg_val:.2f}"        # 4.0 -> "4.00"
    metal_str = f"{metal_val:+.1f}"     # -0.0 -> "-0.0", +1.0 -> "+1.0"
    alpha_str = f"{alpha_val:+.1f}"     # -0.2 -> "-0.2", +0.0 -> "+0.0"

    return f"LTE_T{teff_str}_logg{logg_str}_FeH{metal_str}_alpha{alpha_str}_phoenixMidRes_compressed.txt"


def process_file(infile, outdir, factor):
    # Decide output path
    base = os.path.basename(infile)

    # Prefer PHOENIX → LTE naming
    new_name = build_outname_phoenix(base)
    if new_name is None:
        # Fallback if the file is something else
        root, ext = os.path.splitext(base)
        new_name = f"{root}_compressed{factor}{ext}"

    outfile = os.path.join(outdir, new_name)

    if os.path.isfile(outfile):
        print(f"Skipping {infile} because output file already exists: {outfile}")
    else:
        # Load, header starts with '#'
        with open(infile) as f:
            first = f.readline()
        delim = "\t" if "\t" in first else None
        data = np.loadtxt(infile, comments="#", delimiter=delim)

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

        np.savetxt(
            outfile,
            np.column_stack([
                spec_ds.spectral_axis.to_value(u.nm),
                spec_ds.flux.to_value(u.erg / (u.cm**2 * u.s * u.nm)),
            ]),
            fmt="%.9f\t%.6e",
            header="wavelength (nm)\tflux (erg/cm^2/s/nm)",
            comments="# ",
        )
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
