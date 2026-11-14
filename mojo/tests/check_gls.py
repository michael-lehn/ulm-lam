"""Checks the result of the Mojo GLS (generalized least squares) programs
against NumPy: b = (X^T M^-1 X)^-1 X^T M^-1 y from the binary files that
examples/gls_hand.mojo (or the generated program) writes.

    python3 tests/check_gls.py DIR
"""
import sys

import numpy as np

d = sys.argv[1]
n, m = 2000, 200
X = np.fromfile(f"{d}/X.bin").reshape(n, m)
M = np.fromfile(f"{d}/M.bin").reshape(n, n)
y = np.fromfile(f"{d}/y.bin")
b = np.fromfile(f"{d}/b.bin")
MiX = np.linalg.solve(M, X)
ref = np.linalg.solve(X.T @ MiX, MiX.T @ y)
err = np.abs(b - ref).max() / np.abs(ref).max()
print(f"relative difference to NumPy: {err:.1e}")
print("PASS" if err < 1e-10 else "FAIL")
sys.exit(0 if err < 1e-10 else 1)
