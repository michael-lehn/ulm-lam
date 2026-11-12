"""Linear attention with ulm.lam: the order of a product chosen at run time.

Linear attention (Katharopoulos et al., Transformers are RNNs, ICML 2020)
replaces softmax(Q K^T) V by phi(Q) phi(K)^T V with a feature map phi:

    out = (phi(Q) phi(K)^T V) / (phi(Q) phi(K)^T 1)        (row by row)

Written as the formula, NumPy evaluates from left to right: the N x N
matrix phi(Q) phi(K)^T, O(N^2 d) for a sequence of length N.  Computed as
phi(Q) (phi(K)^T V) it costs O(N d^2), the point of linear attention; the
programmer has to know that and write the parentheses.  ulm.lam chooses
the order at run time from the sizes (here N and d are run-time sizes).
The feature map phi(x) = elu(x) + 1 and the final division are elementwise
and done in NumPy.

    python3 python/examples/attention.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Matrix  # noqa: E402


@lam.function
def attention(Q: Matrix("N", "d"), K: Matrix("N", "d"), V: Matrix("N", "e")):
    # numerator and denominator in one product: V with a column of ones
    return Q @ K.T @ V


def phi(x):
    return np.where(x > 0, x + 1, np.exp(np.minimum(x, 0)))


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    lib = lam.compile(attention, name="attention_py")
    # the generated code, kept in the repository (doc/ulm-lam.tex, appendix)
    (Path(__file__).resolve().parent / "generated" / "attention.cpp").write_text(
        lib.source)
    rng = np.random.default_rng(4)
    d = 64
    print(f"linear attention, head dimension d = {d}; times in s, (x): "
          "relative to ulm.lam")
    print("      N   ulm.lam   NumPy as written      NumPy, parentheses by "
          "hand   difference")
    ok = True
    for N in (256, 1024, 4096, 16384):
        Q, K = phi(rng.normal(size=(N, d))), phi(rng.normal(size=(N, d)))
        V1 = np.hstack([rng.normal(size=(N, d)), np.ones((N, 1))])
        reps = 5 if N <= 4096 else 2
        t_lam, r = best(lambda: lib.attention(Q, K, V1), reps)
        t_lit, _ = best(lambda: Q @ K.T @ V1, 1 if N > 4096 else reps)
        t_hand, ref = best(lambda: Q @ (K.T @ V1), reps)
        out = r[:, :d] / r[:, d:]
        out_ref = ref[:, :d] / ref[:, d:]
        err = np.abs(out - out_ref).max() / np.abs(out_ref).max()
        ok &= err < 1e-12
        print(f"  {N:5d}  {t_lam:7.4f}   {t_lit:7.4f} ({t_lit / t_lam:5.1f}x)"
              f"      {t_hand:7.4f} ({t_hand / t_lam:4.1f}x)        {err:.1e}")
    # short sequences: the other order is the cheaper one, ulm.lam takes it
    Q, K = phi(rng.normal(size=(16, 512))), phi(rng.normal(size=(16, 512)))
    V1 = np.hstack([rng.normal(size=(16, 512)), np.ones((16, 1))])
    t_lam, _ = best(lambda: lib.attention(Q, K, V1), 20)
    t_l, _ = best(lambda: (Q @ K.T) @ V1, 20)
    t_r, _ = best(lambda: Q @ (K.T @ V1), 20)
    print(f"N = 16, d = 512 (the other order is cheaper): ulm.lam {t_lam:.5f} "
          f"s, (Q K^T) V {t_l:.5f} s, Q (K^T V) {t_r:.5f} s")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
