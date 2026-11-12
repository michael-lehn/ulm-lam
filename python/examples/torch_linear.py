"""A PyTorch training loop as formulas: ulm.lam against PyTorch.

A linear layer (torch.nn.Linear: Y = X W + 1 b^T) trained by mini-batch
gradient descent on the mean squared error, the forward pass, the
gradients and the update written as formulas:

    R  = X W + 1 b^T - Y                      forward pass, residual
    W <- W - step X^T R                       step = 2 lr / B
    b <- b - step R^T 1

ulm.lam compiles the whole loop: per step three gemm for the forward pass
(the bias as a rank-1 product), one gemm with alpha = -step and beta = 1
directly into W, one gemv into b; W and b stay in place, no allocation in
the loop; the batches come from Python callbacks.  It does the work of
PyTorch's autograd (the gradient formulas are written here, ulm.lam does
not differentiate) and of the optimizer.

Compared with PyTorch 2.15 on the same BLAS (Accelerate) and in the same
precision (float64): nn.Linear with autograd and optim.SGD, the same steps
written with torch operations without autograd, and those compiled with
torch.compile.

PyTorch brings its own OpenMP runtime, which must not meet Homebrew's
(NumPy's OpenBLAS, ulm.lam's OpenMP) in one process.  Hence a virtual
environment with NumPy and PyTorch from pip (both on Accelerate), ulm.lam
on Accelerate (backend="cblas") and on ulmBLAS without OpenMP:

    python3 -m venv build/venv-torch
    build/venv-torch/bin/pip install numpy torch
    ULM_LAM_BLAS_LIBS="-framework Accelerate" OPENMP=0 \\
        build/venv-torch/bin/python python/examples/torch_linear.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Float, Index, Matrix, Vector  # noqa: E402

batch_x = lam.extern("batch_x", [Index], Matrix("B", "p"))
batch_y = lam.extern("batch_y", [Index], Matrix("B", "k"))


@lam.function
def train(W0: Matrix("p", "k"), b0: Vector("k"), ones: Vector("B"),
          step: Float, steps: Index):
    with lam.loop(steps, carry=[W0, b0]) as L:
        W, b = L.carried
        X, Y = batch_x(L.index), batch_y(L.index)
        R = X @ W + ones @ b.T - Y                      # forward, residual
        L.next(W - step * (X.T @ R), b - step * (R.T @ ones))
    return L.results


def torch_autograd(Xs, Ys, W0, b0, lr):
    """nn.Linear, the mean squared error summed over the outputs, SGD."""
    p, k = W0.shape
    model = torch.nn.Linear(p, k, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(W0.T))
        model.bias.copy_(torch.from_numpy(b0[:, 0]))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for X, Y in zip(Xs, Ys):
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).sum() / X.shape[0]
        loss.backward()
        opt.step()
    return model.weight.detach().T, model.bias.detach()


def torch_by_hand(Xs, Ys, W0, b0, lr):
    """The same steps as torch operations, without autograd."""
    W, b = torch.from_numpy(W0.copy()), torch.from_numpy(b0[:, 0].copy())
    for X, Y in zip(Xs, Ys):
        step = 2 * lr / X.shape[0]
        R = X @ W + b - Y
        W.sub_(step * (X.T @ R))
        b.sub_(step * R.sum(0))
    return W, b


def sgd_step(W, b, X, Y, step):
    R = X @ W + b - Y
    return W - step * (X.T @ R), b - step * R.sum(0)


def torch_compiled(Xs, Ys, W0, b0, lr, compiled):
    W, b = torch.from_numpy(W0.copy()), torch.from_numpy(b0[:, 0].copy())
    for X, Y in zip(Xs, Ys):
        W, b = compiled(W, b, X, Y, 2 * lr / X.shape[0])
    return W, b


def best(f, reps):
    t, r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    torch.set_num_threads(torch.get_num_threads())
    libs = {"Accelerate": lam.compile(train, name="torch_linear_acc",
                                      backend="cblas"),
            "ulmBLAS": lam.compile(train, name="torch_linear_ulm")}
    # the generated code (ulmBLAS), kept in the repository (doc, appendix)
    (Path(__file__).resolve().parent / "generated" / "torch_linear.cpp"
     ).write_text(libs["ulmBLAS"].source)
    try:
        compiled = torch.compile(sgd_step)
    except Exception:  # pragma: no cover - depends on the installation
        compiled = None
    rng = np.random.default_rng(6)
    lr, steps = 0.01, 100
    print(f"linear layer, {steps} steps of mini-batch gradient descent, "
          f"float64, PyTorch {torch.__version__} ({torch.get_num_threads()}"
          " threads); seconds, (x): relative to ulm.lam on Accelerate")
    print("     B      p      k   ulm.lam/Accelerate  ulm.lam/ulmBLAS     "
          "torch autograd    torch by hand  torch.compile   difference")
    ok = True
    for B, p, k in ((32, 64, 16), (256, 1024, 256), (1024, 4096, 1024)):
        Wt = rng.normal(size=(p, k)) / np.sqrt(p)
        Xs = [rng.normal(size=(B, p)) for _ in range(steps)]
        Ys = [X @ Wt + 0.1 * rng.normal(size=(B, k)) for X in Xs]
        tX = [torch.from_numpy(X) for X in Xs]
        tY = [torch.from_numpy(Y) for Y in Ys]
        W0, b0 = np.zeros((p, k)), np.zeros((k, 1))
        ones = np.ones(B)
        reps = 3 if p <= 1024 else 1
        times, res = {}, {}
        for name, lib in libs.items():
            lib.bind(batch_x=lambda i: Xs[i], batch_y=lambda i: Ys[i])
            times[name], res[name] = best(
                lambda: lib.train(W0, b0, ones, 2 * lr / B, steps), reps)
        t_auto, (Wa, ba) = best(lambda: torch_autograd(tX, tY, W0, b0, lr),
                                reps)
        t_hand, _ = best(lambda: torch_by_hand(tX, tY, W0, b0, lr), reps)
        t_comp = float("nan")
        if compiled is not None:
            try:
                torch_compiled(tX, tY, W0, b0, lr, compiled)  # compile once
                t_comp, _ = best(
                    lambda: torch_compiled(tX, tY, W0, b0, lr, compiled), reps)
            except Exception as e:  # pragma: no cover
                print(f"torch.compile failed: {type(e).__name__}")
                compiled = None
        W, b = res["Accelerate"]
        err = max(np.abs(W - Wa.numpy()).max() / np.abs(Wa.numpy()).max(),
                  np.abs(b[:, 0] - ba.numpy()).max() / np.abs(ba.numpy()).max(),
                  np.abs(res["ulmBLAS"][0] - W).max() / np.abs(W).max())
        ok &= err < 1e-10
        t0 = times["Accelerate"]

        def f(t):
            return ("        -       " if t != t
                    else f"{t:7.3f} ({t / t0:4.1f}x)")

        print(f"  {B:4d}  {p:5d}  {k:5d}   {t0:10.3f}        "
              f"{f(times['ulmBLAS'])}  {f(t_auto)}  {f(t_hand)}  {f(t_comp)}"
              f"   {err:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
