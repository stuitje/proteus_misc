#!/usr/bin/env python3
import argparse
import os
import glob
import re
import zipfile

# Match filenames like:
# LTE_T12000_logg6.00_FeH+1.0_alpha+0.0_phoenixMidRes_compressed.txt
PAT_META_ALPHA = re.compile(
    r"^LTE_T\d{5}_logg[0-9.]+_FeH([+-]\d+\.\d)_alpha([+-]\d+\.\d)_phoenixMidRes_compressed\.txt$"
)


def group_files_by_metal_alpha(indir):
    files = sorted(glob.glob(os.path.join(indir, "*.txt")))
    groups = {}

    for path in files:
        base = os.path.basename(path)
        m = PAT_META_ALPHA.match(base)
        if not m:
            # Skip anything that doesn't match the expected pattern
            continue

        feh, alpha = m.groups()   # e.g. "+1.0", "+0.0"
        key = (feh, alpha)
        groups.setdefault(key, []).append(path)

    return groups


def zip_groups(groups, outdir, overwrite=False):
    os.makedirs(outdir, exist_ok=True)

    for (feh, alpha), paths in groups.items():
        # Build a readable zip name, e.g. FeH+1.0_alpha+0.0_phoenixMidRes_compressed.zip
        zip_name = f"FeH{feh}_alpha{alpha}_phoenixMidRes_compressed.zip"
        zip_path = os.path.join(outdir, zip_name)

        if os.path.exists(zip_path) and not overwrite:
            print(f"Skipping {zip_path} (already exists, use --overwrite to replace)")
            continue

        print(f"Creating {zip_path} with {len(paths)} files")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                arcname = os.path.basename(p)  # only the filename inside the zip
                zf.write(p, arcname=arcname)


def main():
    p = argparse.ArgumentParser(
        description="Zip PHOENIX spectra per [Fe/H] + [alpha/Fe] combination."
    )
    p.add_argument(
        "--indir", default="out_compressed",
        help="Directory containing compressed spectra (default: out_compressed/)"
    )
    p.add_argument(
        "--outdir", default="out_zipped_2",
        help="Output directory for ZIP files (default: out_zipped_2/)"
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing ZIP files if they already exist."
    )
    args = p.parse_args()

    groups = group_files_by_metal_alpha(args.indir)
    if not groups:
        print(f"No matching LTE_T*_phoenixMidRes_compressed.txt files found in {args.indir}")
        return

    print("Found the following [Fe/H], [alpha/Fe] groups:")
    for (feh, alpha), paths in sorted(groups.items()):
        print(f"  FeH{feh}, alpha{alpha}: {len(paths)} files")
    print()

    zip_groups(groups, args.outdir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
