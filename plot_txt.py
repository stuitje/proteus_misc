#!/usr/bin/env python3
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description="Overlay two spectra (wavelength vs flux).")
p.add_argument("file_a")
p.add_argument("file_b")
args = p.parse_args()

d1 = np.genfromtxt(args.file_a, names=True)  # reads '# col1 col2' as names
d2 = np.genfromtxt(args.file_b, names=True)

xlab, ylab = d1.dtype.names[:2]              # column names from header
l1 = os.path.splitext(os.path.basename(args.file_a))[0]
l2 = os.path.splitext(os.path.basename(args.file_b))[0]

plt.plot(d1[xlab], d1[ylab], label=l1)
plt.plot(d2[xlab], d2[ylab], label=l2, alpha=0.85)
plt.xlabel(xlab.replace("_"," "))
plt.ylabel(ylab.replace("_"," "))
plt.xlim(657, 658)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
