"""K-FAC as formulas: a second-order optimizer for PyTorch, compiled by ulm.lam.

K-FAC (Kronecker-factored approximate curvature; Martens and Grosse, 2015)
preconditions the gradient of each linear layer with two small matrices
instead of the full Fisher information: with a (B x i) the inputs of the
layer and g (B x o) the gradients of the loss with respect to its outputs,

    A <- rho A + (1 - rho)/B a^T a           running averages (every step)
    G <- rho G + (1 - rho)/B g^T g
    Ai = (A + gamma I)^-1, Gi = (G + gamma I)^-1   refreshed every T steps
    W <- W - lr Gi grad Ai,  grad = g^T a    the preconditioned step

Written as formulas, ulm.lam compiles them to: syrk for the running
averages (one triangle; they are symmetric by construction), potrf and
potri for the damped inverses, and for the step the cheapest order of the
chain Gi g^T a Ai, chosen at run time: with a batch smaller than the layer
(B < i, o), Gi g^T and a Ai are thin, and the o x i gradient is never
multiplied by an n x n inverse.

The comparison is the "inverse" method of kfac-pytorch (Pauloski et al.,
github.com/gpauloski/kfac-pytorch), reproduced here operation by operation
(kfac/layers/inverse.py, base.py, utils.py): the factors as a^T (a / B),
symmetrised; the damped inverses by torch.linalg.inv; the step as
g_inv @ grad @ a_inv, two dense n x n x n products per layer and step.  (Its
defaults also compute the inverses in float32 and clip the step by a KL
(Kullback-Leibler) bound; here everything is float64 and unclipped, so that
both produce the same iterates.)  And an expert version in PyTorch: what a
numerical linear algebra person would write by hand (Cholesky-based
inverses, the thin order of the chain).

Run in the PyTorch environment (see torch_linear.py):

    ULM_LAM_BLAS_LIBS="-framework Accelerate" OPENMP=0 \\
        build/venv-torch/bin/python python/examples/kfac.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ulm import lam  # noqa: E402
from ulm.lam import Float, InOut, Matrix  # noqa: E402


# ---- K-FAC in ulm.lam: the formulas ----------------------------------------
# The factors and their inverses are symmetric: stored as their lower
# triangle (LAPACK's uplo; the upper triangle may hold anything), so that
# syrk, potrf, potri and symm work on one triangle and nothing is mirrored.

def Sym(n):
    return Matrix(n, n, "spd", stored="lower")


@lam.function
def kfac_factors(A: InOut(Sym("i")), G: InOut(Sym("o")), a: Matrix("B", "i"),
                 g: Matrix("B", "o"), rho: Float, c: Float):
    """The running averages (every step)."""
    lam.update(A, rho * A + c * (a.T @ a))
    lam.update(G, rho * G + c * (g.T @ g))


@lam.function
def kfac_refresh(A: Sym("i"), G: Sym("o"), Ai: InOut(Sym("i")),
                 Gi: InOut(Sym("o")), gamma: Float):
    """The damped inverses (every T steps)."""
    lam.update(Ai, lam.assume(A + gamma * lam.eye("i"), "spd").inv)
    lam.update(Gi, lam.assume(G + gamma * lam.eye("o"), "spd").inv)


@lam.function
def kfac_precondition(W: InOut(Matrix("o", "i")), Ai: Sym("i"), Gi: Sym("o"),
                      a: Matrix("B", "i"), g: Matrix("B", "o"), lr: Float):
    """The preconditioned step, grad = g^T a (every step)."""
    lam.update(W, W - lr * (Gi @ g.T @ a @ Ai))


@lam.function
def kfac_step(W: InOut(Matrix("o", "i")), A: InOut(Sym("i")),
              G: InOut(Sym("o")), Ai: Sym("i"), Gi: Sym("o"),
              a: Matrix("B", "i"), g: Matrix("B", "o"), rho: Float, c: Float,
              lr: Float):
    """Both in one call (the steps without a refresh)."""
    lam.update(A, rho * A + c * (a.T @ a))
    lam.update(G, rho * G + c * (g.T @ g))
    lam.update(W, W - lr * (Gi @ g.T @ a @ Ai))


KFAC = [kfac_factors, kfac_refresh, kfac_precondition, kfac_step]


# ---- the same in PyTorch -----------------------------------------------------

def cov(x):
    """kfac-pytorch's get_cov: x^T (x / B), symmetrised."""
    c = x.t() @ (x / x.shape[0])
    return (c + c.t()) / 2.0


class KfacPytorch:
    """kfac-pytorch's inverse method (float64, no KL clipping)."""

    def __init__(self, i, o, rho, gamma):
        self.A, self.G = torch.eye(i, dtype=torch.float64), \
            torch.eye(o, dtype=torch.float64)
        self.rho, self.gamma = rho, gamma

    def factors(self, a, g):
        self.A = self.rho * self.A + (1 - self.rho) * cov(a)
        self.G = self.rho * self.G + (1 - self.rho) * cov(g)

    def refresh(self):
        eye = torch.eye
        self.Ai = torch.linalg.inv(self.A + self.gamma * eye(
            self.A.shape[0], dtype=torch.float64))
        self.Gi = torch.linalg.inv(self.G + self.gamma * eye(
            self.G.shape[0], dtype=torch.float64))

    def step(self, W, a, g, grad, lr):
        W.sub_(lr * (self.Gi @ grad @ self.Ai))


class KfacExpert(KfacPytorch):
    """By hand, as an expert would: Cholesky-based inverses, the thin
    order of the chain (B < i, o), the update in place."""

    def factors(self, a, g):
        B = a.shape[0]
        self.A.mul_(self.rho).addmm_(a.t(), a, alpha=(1 - self.rho) / B)
        self.G.mul_(self.rho).addmm_(g.t(), g, alpha=(1 - self.rho) / B)

    def refresh(self):
        def inv(M):
            L = torch.linalg.cholesky(M + self.gamma * torch.eye(
                M.shape[0], dtype=torch.float64))
            return torch.cholesky_inverse(L)
        self.Ai, self.Gi = inv(self.A), inv(self.G)

    def step(self, W, a, g, grad, lr):
        W.addmm_(self.Gi @ g.t(), a @ self.Ai, alpha=-lr)


class KfacLam:
    """The ulm.lam functions on the same data (NumPy views of the torch
    tensors, no copies)."""

    def __init__(self, lib, i, o, rho, gamma):
        self.lib, self.rho, self.gamma = lib, rho, gamma
        self.A, self.G = np.eye(i), np.eye(o)
        self.Ai, self.Gi = np.empty((i, i)), np.empty((o, o))

    def step(self, t, T, W, a, g, lr):
        """Step t: the factors, every T steps the inverses, the update."""
        lib, c = self.lib, (1 - self.rho) / a.shape[0]
        if t % T == 0:
            lib.kfac_factors(self.A, self.G, a, g, self.rho, c)
            lib.kfac_refresh(self.A, self.G, self.Ai, self.Gi, self.gamma)
            lib.kfac_precondition(W, self.Ai, self.Gi, a, g, lr)
        else:
            lib.kfac_step(W, self.A, self.G, self.Ai, self.Gi, a, g,
                          self.rho, c, lr)


# ---- one layer: the preconditioner alone -----------------------------------

def layer_benchmark(libs, sizes, B, T, rounds=3):
    """Time per step (averaged over T steps and one refresh) of the
    preconditioner of one layer, i = o = n, batch B."""
    rng = np.random.default_rng(1)
    rho, gamma, lr = 0.95, 1e-3, 0.1
    print(f"one linear layer n x n, batch B = {B}, inverses refreshed every "
          f"T = {T} steps; milliseconds per step (factors, refresh / T, "
          "preconditioned step), float64")
    print("      n   ulm.lam/Accelerate  ulm.lam/ulmBLAS*  kfac-pytorch "
          "method   expert by hand   difference")
    ok = True
    for n in sizes:
        a_s = [rng.normal(size=(B, n)) for _ in range(T)]
        g_s = [rng.normal(size=(B, n)) / B for _ in range(T)]
        W0 = rng.normal(size=(n, n)) / np.sqrt(n)
        ta = [torch.from_numpy(x) for x in a_s]
        tg = [torch.from_numpy(x) for x in g_s]
        grads = [g.t() @ a for g, a in zip(tg, ta)]  # from autograd

        def run_torch(cls):
            k = cls(n, n, rho, gamma)
            W = torch.from_numpy(W0.copy())
            for t in range(T):
                k.factors(ta[t], tg[t])
                if t == 0:
                    k.refresh()
                k.step(W, ta[t], tg[t], grads[t], lr)
            return W.numpy()

        def run_lam(lib):
            k = KfacLam(lib, n, n, rho, gamma)
            W = W0.copy()
            for t in range(T):
                k.step(t, T, W, a_s[t], g_s[t], lr)
            return W

        results, times = {}, {}
        cands = {"lam/Accelerate": lambda: run_lam(libs["Accelerate"]),
                 "lam/ulmBLAS": lambda: run_lam(libs["ulmBLAS"]),
                 "pytorch": lambda: run_torch(KfacPytorch),
                 "expert": lambda: run_torch(KfacExpert)}
        for name in cands:
            times[name] = float("inf")
        for _ in range(rounds):  # interleaved: the same machine state
            for name, f in cands.items():
                t0 = time.perf_counter()
                results[name] = f()
                times[name] = min(times[name], time.perf_counter() - t0)
        ref = results["pytorch"]
        errs = {k: np.abs(r - ref).max() / np.abs(ref).max()
                for k, r in results.items()}
        if max(errs.values()) >= 1e-8:
            print("   ", {k: f"{e:.1e}" for k, e in errs.items()})
        err = max(errs.values())
        ok &= err < 1e-8
        t0 = times["lam/Accelerate"]

        def f(name):
            t = times[name]
            return f"{1e3 * t / T:8.2f} ({t / t0:4.1f}x)"

        print(f"  {n:5d}     {f('lam/Accelerate')}   {f('lam/ulmBLAS')}  "
              f"{f('pytorch')}    {f('expert')}   {err:.1e}")
    print("  (*) ulmBLAS on one thread: without OpenMP, whose runtime would "
          "clash with PyTorch's")
    return ok


# ---- training a network ------------------------------------------------------

class Kfac:
    """K-FAC for the bias-free linear layers of a model, as an optimizer:
    hooks capture the inputs a and the per-example output gradients g
    (B times the gradient of the mean loss); step() updates the weights in
    place.  impl: "lam" (the ulm.lam functions, on NumPy views of the
    torch tensors), "pytorch" (kfac-pytorch's method), "expert"."""

    def __init__(self, model, impl, lr, rho=0.95, gamma=1e-2, T=10,
                 lib=None):
        self.layers = [m for m in model.modules()
                       if isinstance(m, torch.nn.Linear)]
        self.impl, self.lr, self.T, self.t, self.lib = impl, lr, T, 0, lib
        self.saved = {}
        for m in self.layers:
            m.register_forward_pre_hook(self._save_input)
            m.register_full_backward_hook(self._save_grad)
            o, i = m.weight.shape
            self.saved[m] = {}
            if impl == "lam":
                self.saved[m]["k"] = KfacLam(lib, i, o, rho, gamma)
            else:
                cls = KfacPytorch if impl == "pytorch" else KfacExpert
                self.saved[m]["k"] = cls(i, o, rho, gamma)

    def _save_input(self, m, inp):
        self.saved[m]["a"] = inp[0].detach()

    def _save_grad(self, m, grad_in, grad_out):
        g = grad_out[0].detach()
        self.saved[m]["g"] = g * g.shape[0]

    @torch.no_grad()
    def step(self):
        for m in self.layers:
            s = self.saved[m]
            k, a, g = s["k"], s["a"], s["g"]
            B = a.shape[0]
            if self.impl == "lam":
                # grad = g^T a / B: the step with lr / B
                k.step(self.t, self.T, m.weight.data.numpy(),
                       a.contiguous().numpy(), g.contiguous().numpy(),
                       self.lr / B)
            else:
                k.factors(a, g)
                if self.t % self.T == 0:
                    k.refresh()
                k.step(m.weight.data, a, g / B, m.weight.grad, self.lr)
        self.t += 1


def mlp(widths, seed):
    torch.manual_seed(seed)
    layers = []
    for i, (p, q) in enumerate(zip(widths, widths[1:])):
        layers.append(torch.nn.Linear(p, q, bias=False, dtype=torch.float64))
        if i < len(widths) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def training(libs, widths, B, steps, T, lr=0.02, gamma=0.1):
    """A teacher network labels random inputs; the student (same widths)
    is trained with cross-entropy.  K-FAC by the three implementations
    (the same iterates), and SGD for the cost of a plain step: what K-FAC
    adds to forward and backward pass is the preconditioner."""
    teacher = mlp(widths, seed=1)
    gen = torch.Generator().manual_seed(2)
    X = torch.randn(steps * B, widths[0], generator=gen, dtype=torch.float64)
    with torch.no_grad():
        Y = teacher(X).argmax(1)
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f"\ntraining an MLP {'-'.join(map(str, widths))} (ReLU, no "
          f"biases), batch {B}, {steps} steps, float64; K-FAC: lr {lr}, "
          f"damping {gamma}, inverses every T = {T} steps")
    print("  optimizer                    ms/step   K-FAC's part   "
          "loss (last 20 steps)")
    runs = [("SGD (lr 0.1)", "sgd"), ("K-FAC, kfac-pytorch method", "pytorch"),
            ("K-FAC, expert by hand", "expert"),
            ("K-FAC, ulm.lam (Accelerate)", "lam")]
    weights, ms = {}, {}
    for label, impl in runs:
        model = mlp(widths, seed=3)
        if impl == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
        else:
            opt = Kfac(model, impl, lr, gamma=gamma, T=T,
                       lib=libs["Accelerate"])
        losses = []
        t0 = time.perf_counter()
        for k in range(steps):
            xb, yb = X[k * B:(k + 1) * B], Y[k * B:(k + 1) * B]
            model.zero_grad(set_to_none=False)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ms[impl] = 1e3 * (time.perf_counter() - t0) / steps
        weights[impl] = [m.weight.detach().clone() for m in model
                         if isinstance(m, torch.nn.Linear)]
        part = "" if impl == "sgd" else \
            f"{ms[impl] - ms['sgd']:8.2f} ms"
        print(f"  {label:28s} {ms[impl]:8.2f}   {part:12s}   "
              f"{np.mean(losses[-20:]):8.4f}")
    base = ms["sgd"]
    print(f"  K-FAC's part, kfac-pytorch method / ulm.lam: "
          f"{(ms['pytorch'] - base) / (ms['lam'] - base):.1f}x, "
          f"expert / ulm.lam: {(ms['expert'] - base) / (ms['lam'] - base):.1f}x")
    diff = max((w - r).abs().max().item() / r.abs().max().item()
               for w, r in zip(weights["lam"], weights["pytorch"]))
    print(f"  K-FAC weights after {steps} steps, ulm.lam against the "
          f"kfac-pytorch method: {diff:.1e} (relative)")
    return diff < 1e-8


def main():
    libs = {"Accelerate": lam.compile(KFAC, name="kfac_acc",
                                      backend="cblas"),
            "ulmBLAS": lam.compile(KFAC, name="kfac_ulm")}
    for f in ("kfac_factors", "kfac_refresh", "kfac_precondition"):
        print(getattr(libs["ulmBLAS"], f).code)
    print(f"PyTorch {torch.__version__}, {torch.get_num_threads()} threads\n")
    ok = layer_benchmark(libs, (512, 1024, 2048), B=128, T=10)
    ok &= training(libs, (256, 1024, 1024, 10), B=128, steps=300, T=10)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
