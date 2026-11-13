"""The e-graph experiment driven by ulm.lam's IR.

A problem is a ulm.lam function written with the Python frontend, with
static sizes (Matrix(2000, 50, "full_rank")).  Its traced ulm_lam IR is

  1. translated into a term of the e-graph of lamp_egraph.py (the same
     constructors, rules and cost model), saturated, and the cheapest plan
     extracted as a DAG;
  2. compiled by ulm.lam's kernel selection (ulm-lam-opt), whose ulm_lamk
     kernels are counted with the same flop formulas (static sizes: one
     plan, no decision at run time).

Where the e-graph finds fewer flops, ulm.lam's selection misses something
(the way the Kalman finding of lamp_egraph.py came about, now for any
function).  Run with the venv of the experiment (make egraph-ir):

    build/venv-egglog/bin/python experiments/egraph/lamp_ir.py [name...]
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from egglog import EGraph  # noqa: E402

import lamp_egraph as G  # noqa: E402
from ulm import lam  # noqa: E402
from ulm.lam import Matrix, Vector  # noqa: E402

OPT = ROOT / "build" / "bin" / "ulm-lam-opt"
SELECT = ("builtin.module(func.func(ulm-lam-normalize, "
          "ulm-lam-infer-properties, ulm-lam-select-kernels), "
          "ulm-lam-strip-properties, canonicalize)")

# -- the IR -------------------------------------------------------------------

_TYPE = re.compile(r"tensor<(\d+)x(\d+)xf64(?:, #ulm_lam\.props<([^>]*)>)?>")


def shape(t: str):
    """(rows, cols, props) of a tensor type."""
    m = _TYPE.match(t.strip())
    if not m:
        raise ValueError(f"not a static matrix type: {t}")
    props = [p.strip() for p in (m.group(3) or "").split(",") if p.strip()]
    return int(m.group(1)), int(m.group(2)), props


def split_types(s: str):
    """Types separated by commas (commas inside <...> do not count)."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "<"
        depth -= ch == ">"
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


_FACT = {"spd": G.spd, "spsd": G.spsd, "symmetric": G.sym,
         "full_rank": G.fullrank, "lower_tri": G.lower}


def ir_to_term(text: str, fname: str):
    """The e-graph term of function fname of the traced ulm_lam IR, and the
    facts its types assert (properties of parameters and of assume)."""
    i = text.index(f"func.func @{fname}(")
    body = text[i:text.index("\n}", i)].split("\n")
    head = body[0]
    vals, consts, facts = {}, {}, []

    def fact(term, props):
        for p in props:
            if p in _FACT:
                facts.append(_FACT[p](term))

    for m in re.finditer(r"(%p\.\w+): (tensor<[^{]*?>)(?= \{|,|\))", head):
        r, c, props = shape(m.group(2))
        name = m.group(1)[3:]
        vals[m.group(1)] = G.Mat(name, r, c)
        fact(vals[m.group(1)], props)
    result = None
    for line in body[1:]:
        line = line.strip()
        if line.startswith("return"):
            outs = [x.strip() for x in line[6:].split(" : ")[0].split(",")]
            terms = [vals[o] for o in outs]
            result = terms[0]
            for t in terms[1:]:
                result = G.Pair(result, t)
            continue
        m = re.match(r"(%[\w.]+) = arith\.constant ([-\d.e+]+)", line)
        if m:
            consts[m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"(%[\w.]+) = ulm_lam\.identity : (.*)$", line)
        if m:
            r, c, _ = shape(m.group(2))
            vals[m.group(1)] = G.Mat("I", r, c)
            facts.append(G.spd(vals[m.group(1)]))
            continue
        m = re.match(r"(%[\w.]+) = ulm_lam\.(\w+) ?([^:]*): (.*) -> (.*)$",
                     line)
        if not m:
            raise ValueError(f"unsupported line: {line}")
        res, op, args = m.group(1), m.group(2), \
            [a.strip() for a in m.group(3).split(",") if a.strip()]
        rtype = m.group(5)
        if op == "transpose":
            vals[res] = G.T(vals[args[0]])
        elif op == "inverse":
            vals[res] = G.Inv(vals[args[0]])
        elif op == "mul":
            t = vals[args[0]]
            for a in args[1:]:
                t = t * vals[a]
            vals[res] = t
        elif op == "add":
            # a term scaled by -1 is subtracted; other scalars are the
            # alpha of a kernel (free)
            t = None
            for a in args:
                v, neg = vals[a], a in negated
                t = v if t is None else (t - v if neg else t + v)
            vals[res] = t
        elif op == "scale":
            alpha, x = args
            vals[res] = vals[x]
            if consts.get(alpha) == -1.0:
                negated.add(res)
        elif op == "identity":
            r, c, _ = shape(rtype)
            vals[res] = G.Mat("I", r, c)
            facts.append(G.spd(vals[res]))
        elif op == "assume":
            vals[res] = vals[args[0]]
            fact(vals[res], shape(rtype)[2])
        else:
            raise ValueError(f"unsupported op ulm_lam.{op}")
    return result, facts


negated: set = set()

# -- ulm.lam's plan -----------------------------------------------------------


def kernel_flops(line: str) -> tuple[str, float]:
    """The kernel of a ulm_lamk line and its flops (the cost model of
    lamp_egraph.py: copies and transposes are free)."""
    op = re.match(r".*ulm_lamk\.(\w+)", line).group(1)
    ins = re.search(r"ins\([^:]*: ([^)]*)\)", line)
    outs = re.search(r"outs\([^:]*: ([^)]*)\)", line)
    def mats(group):  # (pivots, tensor<nxi32>, are not matrices)
        return [shape(t)[:2] for t in split_types(group)
                if "xf64" in t or "xf32" in t]
    it = mats(ins.group(1)) if ins else []
    ot = mats(outs.group(1)) if outs else []
    trans = lambda a: re.search(rf"{a}\(trans\)", line) is not None  # noqa
    if op == "gemm":
        (m, n), (ar, ac) = ot[0], it[0]
        k = ar if trans("trans_a") else ac
        return op, 2.0 * m * n * k
    if op == "gemv":
        r, c = it[0]
        return op, 2.0 * r * c
    if op in ("symm", "symv"):
        (m, n), (k, _) = ot[0], it[0]
        return op, 2.0 * m * n * k
    if op in ("syrk", "gemmtr"):
        (n, _), (ar, ac) = ot[0], it[0]
        k = ar if trans("trans") or trans("trans_a") else ac
        return op, 1.0 * n * n * k
    if op == "potrf" or op == "sytrf":
        n = ot[0][0]
        return op, n ** 3 / 3.0
    if op == "getrf":
        n = ot[0][0]
        return op, 2.0 * n ** 3 / 3.0
    if op in ("trsm", "trmm"):
        m, n = ot[0]
        left = "side(right)" not in line
        return op, 1.0 * m * m * n if left else 1.0 * m * n * n
    if op == "trsv":
        n = ot[0][0]
        return op, 1.0 * n * n
    if op == "getrs":
        n, k = ot[0]
        return op, 2.0 * n * n * k
    if op in ("geaxpy", "gescal"):
        m, n = ot[0]
        return op, 1.0 * m * n
    if op == "adddiag":
        return op, float(ot[0][0])
    if op in ("gecopy", "sycomplete"):
        return op, 0.0
    raise ValueError(f"no flop count for ulm_lamk.{op}")


def ulmlam_plan(text: str, fname: str):
    """ulm.lam's kernels for fname and their flops."""
    out = subprocess.run([str(OPT), f"--pass-pipeline={SELECT}"],
                         input=text, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    sel = out.stdout
    i = sel.index(f"func.func @{fname}(")
    body = sel[i:sel.index("\n  }", i)]
    if "scf.if" in body:
        raise RuntimeError("a decision at run time (dynamic sizes?)")
    kernels = [kernel_flops(line) for line in body.split("\n")
               if "ulm_lamk." in line]
    return kernels, sum(f for _, f in kernels)


# -- the e-graph --------------------------------------------------------------


def egraph_plan(term, facts, iters=12):
    eg = EGraph()
    root = eg.let("root", term)
    for f in facts:
        eg.register(f)
    t0 = time.perf_counter()
    eg.run((G.sizes.saturate() + G.rules) * iters + G.sizes.saturate())
    t = time.perf_counter() - t0
    dag, cost = eg.extract(root, include_cost=True, cost_model=G.dag_cost,
                           extractor="greedy-dag")
    return dag, cost, t


# -- the problems -------------------------------------------------------------
# Linnea's application problems (Barthels, Psarras, Bientinesi, 2021) and
# the showcases of ulm.lam, with static sizes.

PROBLEMS = {}


def problem(fn):
    PROBLEMS[fn.__name__] = lam.function(fn)
    return fn


n, m = 2000, 50


@problem
def gls(X: Matrix(n, m, "full_rank"), M: Matrix(n, n, "spd"),
        y: Vector(n)):
    return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y


@problem
def gp(Ks: Matrix(4000, 1000), Kn: Matrix(4000, 4000, "spd"),
       Kss: Matrix(1000, 1000, "symmetric"), y: Vector(4000)):
    return Ks.T @ Kn.inv @ y, Kss - Ks.T @ Kn.inv @ Ks


@problem
def fpft(F: Matrix(2000, 2000), P: Matrix(2000, 2000, "spd"),
         Q: Matrix(2000, 2000, "spsd")):
    return F @ P @ F.T + Q


@problem
def kalman_update(H: Matrix(100, 1000), P: Matrix(1000, 1000, "spd"),
                  R: Matrix(100, 100, "spd"), z: Vector(100),
                  x: Vector(1000)):
    S = lam.assume(H @ P @ H.T + R, "spd")
    K = P @ H.T @ S.inv
    return x + K @ (z - H @ x), P - K @ H @ P


@problem
def attention(Q: Matrix(4096, 64), K: Matrix(4096, 64),
              V: Matrix(4096, 64)):
    return Q @ K.T @ V


@problem
def ridge(A: Matrix(3000, 200, "full_rank"), b: Vector(3000)):
    return (A.T @ A + lam.eye(200)).inv @ A.T @ b


@problem
def normal_equations(A: Matrix(3000, 200, "full_rank"), b: Vector(3000)):
    return (A.T @ A).inv @ A.T @ b


@problem
def optimization(W: Matrix(1000, 1000, "spd"), A: Matrix(200, 1000,
                                                           "full_rank"),
                 b: Vector(200), x: Vector(1000)):
    # Linnea's "optimization problem": x + W A^T (A W A^T)^-1 (b - A x)
    S = lam.assume(A @ W @ A.T, "spd")
    return x + W @ A.T @ S.inv @ (b - A @ x)


@problem
def ensemble_kalman(B: Matrix(1000, 1000, "spd"), H: Matrix(200, 1000),
                    R: Matrix(200, 200, "spd"), Y: Matrix(200, 50),
                    Xb: Matrix(1000, 50)):
    S = lam.assume(H @ B @ H.T + R, "spd")
    return Xb + B @ H.T @ S.inv @ (Y - H @ Xb)


@problem
def image_restoration(H: Matrix(2000, 1000, "full_rank"), y: Vector(2000),
                      v: Vector(1000)):
    # (H^T H + I)^-1 (H^T y + v)
    return (H.T @ H + lam.eye(1000)).inv @ (H.T @ y + v)


@problem
def two_solves(A: Matrix(1500, 1500, "non_singular"), B: Matrix(1500, 1500,
                                                               "spd"),
               c: Vector(1500)):
    # A^-1 B^-1 c and A^-1 c: the factorisation of A is shared
    return A.inv @ B.inv @ c, A.inv @ c


@problem
def triangular_sandwich(L: Matrix(2000, 2000, "lower_tri"),
                        A: Matrix(2000, 2000, "spd")):
    return L.inv @ A @ L.inv.T


@problem
def chain_vector(A: Matrix(1000, 2000), B: Matrix(2000, 1500),
                 C: Matrix(1500, 1000), v: Vector(1000)):
    return A @ B @ C @ v


@problem
def gram_of_solve(L: Matrix(2000, 2000, "spd"), X: Matrix(2000, 100)):
    # X^T L^-1 X with L spd: (C^-1 X)^T (C^-1 X), C = chol(L)
    return X.T @ L.inv @ X


@problem
def kalman_gain_cov(P: Matrix(1500, 1500, "spd"), H: Matrix(300, 1500),
                    R: Matrix(300, 300, "spd")):
    # the covariance update alone, P - P H^T (H P H^T + R)^-1 H P
    S = lam.assume(H @ P @ H.T + R, "spd")
    return P - P @ H.T @ S.inv @ H @ P


def main(names):
    rows = []
    for name, fn in PROBLEMS.items():
        if names and name not in names:
            continue
        global negated
        negated = set()
        text = lam.module([fn])
        print(f"\n== {name}")
        try:
            kernels, ulm = ulmlam_plan(text, name)
        except Exception as ex:
            print("   ulm.lam:", str(ex).strip().split("\n")[0][:200])
            continue
        print("   ulm.lam:  %.3g flops  %s" % (
            ulm, " ".join(k for k, _ in kernels)))
        try:
            term, facts = ir_to_term(text, name)
            dag, cost, t = egraph_plan(term, facts)
        except Exception as ex:
            print("   e-graph:", type(ex).__name__, str(ex)[:200])
            continue
        print("   e-graph:  %.3g flops  (saturation %.2f s)" % (cost, t))
        if cost < 0.999 * ulm:
            print("   plan:    ", str(dag).replace("\n", " ")[:600])
        rows.append((name, ulm, cost))
    print("\n| problem | ulm.lam | e-graph (DAG) | e-graph / ulm.lam |")
    print("|:--|--:|--:|--:|")
    for name, ulm, eg in rows:
        print(f"| {name} | {ulm:.3g} | {eg:.3g} | {eg / ulm:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
