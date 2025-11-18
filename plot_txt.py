#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import matplotlib.pyplot as plt

def parse_teff_from_name(path):
    m = re.search(r"lte(\d{5})", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else None

def planck_lambda_nm(T, lam_nm):
    # Planck B_lambda (SI units W sr^-1 m^-3), converted to per-nm (×1e-9).
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    lam_m = lam_nm * 1e-9
    x = (h * c) / (lam_m * k * T)
    # exp(x) might overflow; clipped x
    x = np.clip(x, 1e-9, 700.0)
    B = (2.0 * h * c**2) / (lam_m**5 * (np.exp(x) - 1.0))
    return B * 1e-9

def fit_scale(model, data):
    # least-squares scale a that minimizes ||a*model - data||^2
    m = np.isfinite(model) & np.isfinite(data)
    if not np.any(m):
        return 1.0
    num = np.dot(model[m], data[m])
    den = np.dot(model[m], model[m])
    return num / den if den > 0 else 1.0

p = argparse.ArgumentParser(description="Overlay two spectra (wavelength vs flux).")
p.add_argument("file_a")
p.add_argument("file_b")
args = p.parse_args()

d1 = np.genfromtxt(args.file_a, names=True, delimiter="\t", dtype=None, encoding=None)
d2 = np.genfromtxt(args.file_b, names=True, delimiter="\t", dtype=None, encoding=None)


xlab, ylab = d1.dtype.names[:2]              # column names from header
l1 = os.path.splitext(os.path.basename(args.file_a))[0]
l2 = os.path.splitext(os.path.basename(args.file_b))[0]

plt.plot(d1[xlab], d1[ylab], label=l1)
plt.plot(d2[xlab], d2[ylab], label=l2, alpha=0.85)

T1 = parse_teff_from_name(l1)
if T1:
    bb1 = planck_lambda_nm(T1, d1[xlab].astype(float))
    scale = fit_scale(bb1, d1[ylab])
    plt.plot(d1[xlab], bb1*scale, label=f"BB at {T1} K")

T2 = parse_teff_from_name(l2)
if T2:
    bb2 = planck_lambda_nm(T2, d2[xlab].astype(float))
    scale = fit_scale(bb2, d2[ylab])
    plt.plot(d2[xlab], bb2*scale, label=f"BB at {T2} K")

plt.xlabel(xlab.replace("_"," "))
plt.ylabel(ylab.replace("_"," "))
#plt.xlim(657, 658) # for H alpha line
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
