# gls_driver.mojo: runs the generated GLS (generalized least squares)
# program (module gls: ulm-lam --emit=mojo examples/gls/02-normal.mlir) on
# deterministic data and writes X, M, y and b for tests/check_gls.py.  It
# calls the wrapper gls_keep, which leaves X, M and y as they are (gls itself
# overwrites them, as a LAPACK routine).

from std.sys import argv
from std.ffi import external_call
from ulm_lam import kernels as K
from ulm_lam.kernels import P
from gls import gls_keep


# ---- the test driver -------------------------------------------------------


def write(var path: String, p: P[DType.float64], count: Int):
    var c = path.as_c_string_span()
    var mode = String("wb")
    var cm = mode.as_c_string_span()
    var f = external_call["fopen", Int](c.ptr(), cm.ptr())
    _ = external_call["fwrite", Int](p, 8, count, f)
    _ = external_call["fclose", Int32](f)


def main() raises:
    var dir = String(argv()[1]) if len(argv()) > 1 else String(".")
    comptime n = 2000
    comptime m = 200
    var X = List[Float64](length=n * m, fill=0.0)
    var M = List[Float64](length=n * n, fill=0.0)
    var y = List[Float64](length=n, fill=0.0)
    var b = List[Float64](length=m, fill=0.0)
    # deterministic data: X full rank, M = B B^T / n + I with B random
    var s: UInt64 = 12345

    def rnd(mut s: UInt64) -> Float64:
        s = s * 6364136223846793005 + 1442695040888963407
        return Float64(Int(s >> 11)) / Float64(1 << 53) * 2.0 - 1.0

    for i in range(n * m):
        X[i] = rnd(s)
    for i in range(n):
        y[i] = rnd(s)
    var B = List[Float64](length=n * n, fill=0.0)
    for i in range(n * n):
        B[i] = rnd(s)
    var pB = B.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var pM = M.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    # M := B B^T / n + I (row-major; full matrix: the lower triangle is read)
    K.sylrk(n, n, Float64(1.0) / Float64(n), pB, n, 1, Float64(0.0), pM, n, 1)
    for i in range(n):
        for j in range(i + 1, n):
            M[i * n + j] = M[j * n + i]
        M[i * n + i] += 1.0
    _ = B^

    var pX = X.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var py = y.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var pb = b.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var info = List[Int64](length=1, fill=0)
    var pinfo = info.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var w = List[Int8](length=1, fill=0)
    var pw = w.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    # workspace query, then the call with the workspace
    info[0] = -1
    _ = gls_keep(pX, pM, py, pb, pw, pinfo)
    var size = Int(info[0])
    var work = List[Int8](length=size, fill=0)
    var pwork = work.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    info[0] = Int64(size)
    var r = gls_keep(pX, pM, py, pb, pwork, pinfo)
    print(
        "gls_keep: info",
        r,
        " workspace",
        size,
        "bytes  b[0:3] =",
        b[0],
        b[1],
        b[2],
    )
    write(dir + "/X.bin", pX, n * m)
    write(dir + "/M.bin", pM, n * n)
    write(dir + "/y.bin", py, n)
    write(dir + "/b.bin", pb, m)
    _ = X^
    _ = M^
    _ = y^
    _ = b^
    _ = work^
    _ = info^
    _ = w^
