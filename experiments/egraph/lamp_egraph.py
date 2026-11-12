"""Equality saturation for the Linear Algebra Mapping Problem: a first
experiment with egglog (https://github.com/egraphs-good/egglog-python).

The algebra of ulm_lam (products, sums, transposes, inverses) and the
kernels of ulm_lamk (Cholesky, LU, triangular solves and products, syrk)
are constructors of one e-graph; general rules relate them (associativity,
transposes, S = L L^T for spd S, Z^T Z = syrk(Z), solves for products with
inverses), and properties are relations derived by rules (spd, full rank,
symmetric, triangular).  After saturation the cheapest term is extracted
with ulm.lam's flop counts, for the sizes of the problem: as a tree (every
use of a subterm is paid) and as a DAG (a subterm used twice, such as a
factorisation, is paid once).

The flops are compared with what ulm.lam's kernel selection (strategy=gmc:
normal form, hand-written rules, the generalised matrix chain) generates for
the same problems.

    python3.12 -m venv build/venv-egglog
    build/venv-egglog/bin/pip install egglog
    build/venv-egglog/bin/python experiments/egraph/lamp_egraph.py
"""

from __future__ import annotations

import sys
import time

from egglog import (EGraph, Expr, String, StringLike, i64, i64Like, eq,
                    function, relation, rewrite, rule, ruleset, set_, vars_,
                    get_callable_args, get_callable_fn, get_literal_value,
                    DagCostModel)


class Mat(Expr):
    """A matrix: a leaf (name, rows, cols), or an operation."""

    def __init__(self, name: StringLike, r: i64Like, c: i64Like) -> None: ...

    def __mul__(self, other: Mat) -> Mat: ...

    def __add__(self, other: Mat) -> Mat: ...

    def __sub__(self, other: Mat) -> Mat: ...


# the algebra
@function
def T(a: Mat) -> Mat: ...


@function
def Inv(a: Mat) -> Mat: ...


# the kernels (each a value: the factor, or the result of the kernel)
@function
def Chol(s: Mat) -> Mat:
    """L with S = L L^T (potrf)."""


@function
def Lu(a: Mat) -> Mat:
    """The LU factors of A (getrf)."""


@function
def TrsmL(l: Mat, b: Mat) -> Mat:
    """L^-1 B (trsm, trsv)."""


@function
def TrsmLt(l: Mat, b: Mat) -> Mat:
    """L^-T B."""


@function
def LuSolve(f: Mat, b: Mat) -> Mat:
    """A^-1 B with the LU factors of A (getrs)."""


@function
def Syrk(z: Mat) -> Mat:
    """Z^T Z (syrk: one triangle)."""


@function
def Trmm(a: Mat, l: Mat) -> Mat:
    """A L with L triangular (trmm, half the flops of gemm)."""


@function
def Gemmtr(a: Mat, b: Mat) -> Mat:
    """A B known to be symmetric: one triangle (gemmtr, half the flops)."""


@function
def Pair(a: Mat, b: Mat) -> Mat:
    """Two results of one program (costs nothing)."""


@function
def R(a: Mat) -> i64:
    """Rows (derived by rules from the leaves)."""


@function
def C(a: Mat) -> i64:
    """Columns."""


spd = relation("spd", Mat)
spsd = relation("spsd", Mat)
sym = relation("sym", Mat)
fullrank = relation("fullrank", Mat)
lower = relation("lower", Mat)

a, b, c, s, x, z, l, w, e = vars_("a b c s x z l w e", Mat)
(nm,) = vars_("nm", String)
r, k = vars_("r k", i64)

# the sizes of every term, from its operands (bound in the query: egglog
# reads function values only there)
def _size(term, rows_of, cols_of):
    """R(term) = R or C of an operand, C(term) likewise."""
    (ra, ca) = rows_of, cols_of
    return rule(eq(e).to(term), eq(r).to(ra), eq(k).to(ca)).then(
        set_(R(e)).to(r), set_(C(e)).to(k))


sizes = ruleset(
    rule(eq(e).to(Mat(nm, r, k))).then(set_(R(e)).to(r), set_(C(e)).to(k)),
    _size(a * b, R(a), C(b)),
    _size(a + b, R(a), C(a)),
    _size(a - b, R(a), C(a)),
    _size(Gemmtr(a, b), R(a), C(b)),
    _size(T(a), C(a), R(a)),
    _size(Inv(a), R(a), C(a)),
    _size(Chol(a), R(a), C(a)),
    _size(Lu(a), R(a), C(a)),
    _size(TrsmL(l, b), R(b), C(b)),
    _size(TrsmLt(l, b), R(b), C(b)),
    _size(LuSolve(l, b), R(b), C(b)),
    _size(Syrk(z), C(z), C(z)),
    _size(Trmm(a, l), R(a), C(l)),
    _size(Pair(a, b), R(a), C(a)),
)

rules = ruleset(
    # associativity
    rewrite((a * b) * c).to(a * (b * c)),
    rewrite(a * (b * c)).to((a * b) * c),
    # transposes
    rewrite(T(T(a))).to(a),
    rewrite(T(a * b)).to(T(b) * T(a)),
    rewrite(T(b) * T(a)).to(T(a * b)),
    rewrite(T(s)).to(s, spd(s)),
    rewrite(Inv(T(a))).to(T(Inv(a))),
    # inverses become solves: Cholesky for spd matrices, LU otherwise
    rewrite(Inv(s) * b).to(TrsmLt(Chol(s), TrsmL(Chol(s), b)), spd(s)),
    rewrite(Inv(a) * b).to(LuSolve(Lu(a), b)),
    # a solve from the right: B A^-1 = (A^-T B^T)^T
    rewrite(b * Inv(a)).to(T(Inv(T(a)) * T(b))),
    # X^T L^-T = (L^-1 X)^T: a solve moves to the other factor
    rewrite(T(x) * TrsmLt(l, w)).to(T(TrsmL(l, x)) * w),
    # Gram matrices: Z^T Z, and W W^T = (W^T)^T W^T
    rewrite(T(z) * z).to(Syrk(z)),
    rewrite(w * T(w)).to(Syrk(T(w))),
    # A S A^T with S symmetric is symmetric: the last product one triangle
    rewrite((a * s) * T(a)).to(Gemmtr(a * s, T(a)), sym(s)),
    # S = L L^T for spd S, and A L with L triangular
    rewrite(s).to(Chol(s) * T(Chol(s)), spd(s)),
    rewrite(a * l).to(Trmm(a, l), lower(l)),
    # properties
    rule(spd(s)).then(lower(Chol(s)), sym(s)),
    rule(eq(e).to((a * s) * T(a)), spd(s)).then(spsd(e), sym(e)),
    rule(eq(e).to(a * (s * T(a))), spd(s)).then(spsd(e), sym(e)),
    rule(eq(e).to(b + c), spsd(b), spd(c)).then(spd(e)),
    rule(fullrank(x), Chol(s)).then(fullrank(TrsmL(Chol(s), x))),
    rule(fullrank(z), Syrk(z)).then(spd(Syrk(z))),
    rule(fullrank(z), T(z) * z).then(spd(T(z) * z)),
)

INF = 1e30


def rc(egraph, v) -> tuple[int, int]:
    """Rows and columns of the e-class v (looked up in the e-graph)."""
    return (get_literal_value(egraph.lookup_function_value(R(v))),
            get_literal_value(egraph.lookup_function_value(C(v))))


def flops(egraph, e) -> float:
    """The flops of the kernel at the root of a term (ulm.lam's counts),
    from the sizes of its operands."""
    if get_callable_fn(e) is None:
        return 0.0  # a literal (a name, a size)
    f, args = str(get_callable_fn(e)), get_callable_args(e)
    if f in ("Mat", "T", "Pair"):
        return 0.0  # transposes are strides
    if f == "Inv":
        return INF  # never formed: forces solves
    if f == "· * ·":
        (m, k), (_, n) = rc(egraph, args[0]), rc(egraph, args[1])
        return 2.0 * m * k * n
    if f in ("· + ·", "· - ·"):
        m, n = rc(egraph, args[0])
        return 1.0 * m * n
    if f == "Gemmtr":
        (m, k), _ = rc(egraph, args[0]), None
        return 1.0 * m * m * k
    if f == "Chol":
        n = rc(egraph, args[0])[0]
        return n ** 3 / 3.0
    if f == "Lu":
        n = rc(egraph, args[0])[0]
        return 2.0 * n ** 3 / 3.0
    if f in ("TrsmL", "TrsmLt"):
        n, k = rc(egraph, args[1])
        return 1.0 * n * n * k
    if f == "LuSolve":
        n, k = rc(egraph, args[1])
        return 2.0 * n * n * k
    if f == "Syrk":
        k, n = rc(egraph, args[0])
        return 1.0 * n * n * k
    if f == "Trmm":
        (m, n), _ = rc(egraph, args[0]), None
        return 1.0 * m * n * n
    raise ValueError(f)


def tree_cost(egraph, e, children):
    try:
        return flops(egraph, e) + sum(children)
    except Exception as ex:  # (a missing size would be a modelling error)
        print("cost model:", type(ex).__name__, ex, str(e)[:120])
        raise


dag_cost = DagCostModel(flops, 0.0)


def solve(name, term, facts, gmc_flops, iters=12):
    """Saturate, extract as a tree and as a DAG, compare with gmc."""
    eg = EGraph()
    root = eg.let("root", term)
    for f in facts:
        eg.register(f)
    t0 = time.perf_counter()
    eg.run((sizes.saturate() + rules) * iters + sizes.saturate())
    t_sat = time.perf_counter() - t0
    tree, tc = eg.extract(root, include_cost=True, cost_model=tree_cost)
    dag, dc = eg.extract(root, include_cost=True, cost_model=dag_cost,
                         extractor="greedy-dag")
    nodes = sum(eg.all_function_sizes()[i][1]
                for i in range(len(eg.all_function_sizes())))
    print(f"\n== {name}")
    print(f"   saturation {t_sat:.2f} s, {nodes} e-nodes and table rows")
    print(f"   tree extraction: {tc:.3g} flops   {tree}")
    print(f"   DAG extraction:  {dc:.3g} flops   {dag}")
    print(f"   ulm.lam (gmc):   {gmc_flops:.3g} flops")
    return tc, dc, gmc_flops


def main():
    results = []

    # GLS: (X^T M^-1 X)^-1 X^T M^-1 y, M spd, X full rank
    n, m = 2000, 50
    X, M, y = Mat("X", n, m), Mat("M", n, n), Mat("y", n, 1)
    gls = Inv(T(X) * Inv(M) * X) * (T(X) * (Inv(M) * y))
    gmc = n ** 3 / 3 + n * n * m + m * m * n + m ** 3 / 3 + n * n + 2 * n * m \
        + 2 * m * m
    results.append(("GLS", *solve("GLS, n = 2000, m = 50", gls,
                                  [spd(M), fullrank(X)], gmc)))

    # GP: mean K*^T Kn^-1 y and covariance K** - K*^T Kn^-1 K*, Kn spd
    n, k = 4000, 1000
    Ks, Kn, Kss, y = Mat("Ks", n, k), Mat("Kn", n, n), Mat("Kss", k, k), \
        Mat("y", n, 1)
    gp = Pair(T(Ks) * (Inv(Kn) * y), Kss - T(Ks) * (Inv(Kn) * Ks))
    gmc = n ** 3 / 3 + 2 * n * n + 2 * n * k + n * n * k + k * k * n
    results.append(("GP", *solve("GP, n = 4000, m = 1000", gp, [spd(Kn)],
                                 gmc)))

    # F P F^T with P spd (n = 3000): Linnea's trick from general rules?
    n = 3000
    F, P = Mat("F", n, n), Mat("P", n, n)
    fpf = F * P * T(F)
    # n + 2400 < 3 n: ulm.lam's Cholesky rule (potrf, trmm, syrk)
    gmc = n ** 3 / 3 + n ** 3 + n ** 3
    results.append(("FPF^T", *solve("F P F^T, P spd, n = 3000", fpf,
                                    [spd(P)], gmc)))

    # one step of the Kalman filter (predicted P given, P spd): S, K and the
    # covariance update, with H P shared
    n, m = 1000, 100
    H, P, R_ = Mat("H", m, n), Mat("P", n, n), Mat("R", m, m)
    S = H * P * T(H) + R_
    K = P * T(H) * Inv(S)
    kal = Pair(K, P - K * (H * P))
    # ulm.lam: H P (gemm), gemmtr for S, potrf, two trsm for K^T, gemm
    gmc = 2.0 * m * n * n + m * m * n + m ** 3 / 3 + 2.0 * m * m * n \
        + 2.0 * n * m * n
    results.append(("Kalman step", *solve("Kalman step, n = 1000, m = 100",
                                          kal, [spd(P), spd(R_)], gmc)))

    # a matrix chain Q K^T V (linear attention, N = 4096, d = e = 64)
    N, d = 4096, 64
    Q, K, V = Mat("Q", N, d), Mat("K", N, d), Mat("V", N, d)
    att = Q * T(K) * V
    gmc = 2.0 * d * N * d + 2.0 * N * d * d
    results.append(("attention", *solve("Q K^T V, N = 4096, d = 64", att,
                                        [], gmc)))

    print("\nflops        e-graph, tree  e-graph, DAG  ulm.lam (gmc)")
    for name, tc, dc, gc in results:
        print(f"{name:14s}  {tc:10.3g}  {dc:12.3g}  {gc:12.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
