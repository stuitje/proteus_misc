from astropy.io import fits
import numpy as np
import argparse
import os

def open_stellar_fits(filepath, wavelengthpath, output_path):
    with fits.open(filepath) as hdul:
        print(f"\nOpened FITS file: {filepath}")
        print(f"Contains {len(hdul)} HDU(s):\n")

        for i, hdu in enumerate(hdul):
            print(f"  ➤ HDU {i}: {type(hdu).__name__}, shape={getattr(hdu.data, 'shape', None)}")

        primary_header_star = hdul[0].header
        data_star = hdul[0].data

        print("\nPrimary Header Keys:")
        print(list(primary_header_star.keys())[:10])

        if data_star is None:
            print("No data found in the primary HDU.")
            return

        if output_path is None:
            output_path = filepath.replace(".fits", ".txt")

    with fits.open(wavelengthpath) as hdul:
        print(f"\nOpened FITS file: {wavelengthpath}")
        print(f"Contains {len(hdul)} HDU(s):\n")

        for i, hdu in enumerate(hdul):
            print(f"  ➤ HDU {i}: {type(hdu).__name__}, shape={getattr(hdu.data, 'shape', None)}")

        primary_header_wave = hdul[0].header
        data_wave = hdul[0].data

        print("\nPrimary Header Keys:")
        print(list(primary_header_wave.keys())[:10])

        if data_wave is None:
            print("No data found in the primary HDU.")
            return

    label = os.path.basename(filepath).split("HiRes")[-1].split(".fits")[0].strip("_")

    # Convert flux from ergs/cm²/s/cm → ergs/cm²/s/nm, wavelength armstrong -> nm
    combined = np.column_stack((data_wave * 0.1, data_star * 1e-7))

    header = f"wavelength_nm\tflux_erg_cm^-2_s^-1_nm^-1"
    np.savetxt(output_path, combined, fmt="%.6e", delimiter="\t", header=header, comments="# ")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save FITS data to a text file with unit conversion")
    parser.add_argument("file1", help="Path to the stellar spectrum fits file")
    parser.add_argument("file2", help="Path to the wavelength fits file")
    parser.add_argument("--out", help="Optional output .txt filename")

    args = parser.parse_args()
    open_stellar_fits(args.file1, args.file2, args.out)
