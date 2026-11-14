# main.mojo: the example of the lectures, run with the code ulm.lam
# generated (sensor_lam.mojo, from gen.py) and with the formula as written.
#
#   for l in range(steps):          the program, as in gen.py
#       b = get_sensor_data(l)
#       x = inv(A) * b
#       react(l, x)
#
# The sensor: a true state x_l (a slowly changing signal), measured as
# b = A x_l with a general (nonsymmetric) n x n matrix A.  react checks that
# x is x_l.  "As written" computes inv(A) in every step (getrf, getri) and
# multiplies (gemv): O(n^3) per step.  ulm.lam's code factorizes A once and
# solves in place in every step: O(n^2) per step.
#
#   pixi run sensor     (in mojo/: generates sensor_lam.mojo, builds, runs)

from std.math import abs, sin
from std.sys import argv
from std.time import perf_counter_ns
from ulm_lam import kernels as K
from ulm_lam.kernels import P
from sensor_lam import Externs, control


comptime Ptr = P[DType.float64]


def ptr(mut v: List[Float64]) -> Ptr:
    return v.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()


def truth(l: Int, i: Int) -> Float64:
    """The true state x_l, component i."""
    return sin(0.01 * Float64(l) + 0.1 * Float64(i))


struct Sensor(Externs):
    """The application: measures b = A x_l, checks the x it gets back."""

    var A: List[Float64]  # the measurement matrix (row-major)
    var x: List[Float64]  # the true state of the current step
    var err: Float64  # the largest error |x - x_l| so far
    var reacted: Int

    def __init__(out self, var A: List[Float64], n: Int):
        self.A = A^
        self.x = List[Float64](length=n, fill=0.0)
        self.err = 0.0
        self.reacted = 0

    def get_sensor_data(mut self, n: Int, l: Int, b: Ptr):
        for i in range(n):
            self.x[i] = truth(l, i)
        K.gemv(
            n,
            n,
            Float64(1.0),
            ptr(self.A),
            n,
            1,
            ptr(self.x),
            1,
            Float64(0.0),
            b,
            1,
        )

    def react(mut self, n: Int, l: Int, x: Ptr):
        for i in range(n):
            self.err = max(self.err, abs(x[unsafe_offset=i] - truth(l, i)))
        self.reacted += 1


def as_written(mut s: Sensor, n: Int, A: Ptr, steps: Int):
    """The formula as written: inv(A) and the product, in every step."""
    var Ai = List[Float64](length=n * n, fill=0.0)
    var b = List[Float64](length=n, fill=0.0)
    var x = List[Float64](length=n, fill=0.0)
    var piv = List[Int32](length=n, fill=0)
    var pp = piv.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for l in range(steps):
        s.get_sensor_data(n, l, ptr(b))
        K.copy(ptr(Ai), A, 8 * n * n)
        _ = K.getrf(n, n, ptr(Ai), n, 1, pp, 1)
        _ = K.getri(n, ptr(Ai), n, 1, pp, 1)
        K.gemv(
            n,
            n,
            Float64(1.0),
            ptr(Ai),
            n,
            1,
            ptr(b),
            1,
            Float64(0.0),
            ptr(x),
            1,
        )
        s.react(n, l, ptr(x))
    _ = Ai^
    _ = b^
    _ = x^
    _ = piv^


def main() raises:
    var n = Int(argv()[1]) if len(argv()) > 1 else 1000
    var steps = Int(argv()[2]) if len(argv()) > 2 else 200
    # A: random, plus n on the diagonal (well conditioned, not symmetric)
    var A = List[Float64](length=n * n, fill=0.0)
    var r: UInt64 = 7
    for k in range(n * n):
        r = r * 6364136223846793005 + 1442695040888963407
        A[k] = Float64(Int(r >> 11)) / Float64(1 << 53) * 2.0 - 1.0
    for i in range(n):
        A[i * n + i] += Float64(n)
    var pA = ptr(A)
    var s = Sensor(A.copy(), n)

    # the generated code: workspace query (info = -1), then the calls
    # with the workspace (allocated once, here)
    var info = List[Int64](length=1, fill=-1)
    var pinfo = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var none = List[Int8](length=1, fill=0)
    var pnone = none.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _ = control(s, n, pA, steps, pnone, pinfo)
    var work = List[Int8](length=Int(info[0]), fill=0)
    var pwork = work.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    var t0 = perf_counter_ns()
    var r1 = control(s, n, pA, steps, pwork, pinfo)
    var t_lam = Float64(perf_counter_ns() - t0) * 1e-6
    var err_lam = s.err
    var count = s.reacted

    s.err = 0.0
    t0 = perf_counter_ns()
    as_written(s, n, pA, steps)
    var t_lit = Float64(perf_counter_ns() - t0) * 1e-6

    print("n =", n, " steps =", steps, " (Apple Accelerate)")
    print(
        "  ulm.lam (getrf once, getrs per step) ",
        t_lam,
        "ms  info",
        r1,
        " max error",
        err_lam,
    )
    print(
        "  as written (inv(A) * b per step)     ",
        t_lit,
        "ms",
        "             max error",
        s.err,
    )
    print("  speedup", t_lit / t_lam)
    var ok = r1 == 0 and count == steps and err_lam < 1e-10 and s.err < 1e-10
    print("PASS" if ok else "FAIL")
    _ = A^
    _ = info^
    _ = none^
    _ = work^
