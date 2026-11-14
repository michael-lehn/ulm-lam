# test_kernels.mojo: the kernels of ulm_lam.kernels (Apple Accelerate)
# against plain loops, for float64 and float32, on row-major and
# column-major views.  Triangles a routine must not read hold garbage.
#
#   pixi run mojo build -I . -Xlinker -framework -Xlinker Accelerate \
#       tests/test_kernels.mojo -o build/test_kernels && build/test_kernels

from std.math import abs
from ulm_lam import kernels as K
from ulm_lam.kernels import P


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def next(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64(Int(self.s >> 11)) / Float64(1 << 53) * 2.0 - 1.0


struct Mat[T: DType](Movable):
    """A rows x cols matrix, row-major (col = False) or column-major."""

    var data: List[Scalar[Self.T]]
    var rows: Int
    var cols: Int
    var col: Bool

    def __init__(out self, rows: Int, cols: Int, col: Bool):
        self.data = List[Scalar[Self.T]](length=max(rows * cols, 1), fill=0)
        self.rows = rows
        self.cols = cols
        self.col = col

    def incRow(self) -> Int:
        return 1 if self.col else self.cols

    def incCol(self) -> Int:
        return self.rows if self.col else 1

    def ptr(mut self) -> P[Self.T]:
        return self.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

    def get(self, i: Int, j: Int) -> Float64:
        return Float64(self.data[i * self.incRow() + j * self.incCol()])

    def set(mut self, i: Int, j: Int, v: Float64):
        self.data[i * self.incRow() + j * self.incCol()] = Scalar[Self.T](v)

    def fill(mut self, mut r: Rng):
        for i in range(self.rows):
            for j in range(self.cols):
                self.set(i, j, r.next())

    def copy(self) -> Self:
        var m = Self(self.rows, self.cols, self.col)
        for k in range(len(self.data)):
            m.data[k] = self.data[k]
        return m^


struct Check(Movable):
    var failed: Int
    var tol: Float64

    def __init__(out self, tol: Float64):
        self.failed = 0
        self.tol = tol

    def near(mut self, what: String, err: Float64):
        if not (err <= self.tol):
            print("FAIL", what, "error", err)
            self.failed += 1


def ori(c: Bool) -> String:
    return "c" if c else "r"


def spd[T: DType](n: Int, col: Bool, mut r: Rng) -> Mat[T]:
    """B B^T / n + I."""
    var B = Mat[T](n, n, True)
    B.fill(r)
    var A = Mat[T](n, n, col)
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for k in range(n):
                s += B.get(i, k) * B.get(j, k)
            A.set(i, j, s / Float64(n) + (1.0 if i == j else 0.0))
    return A^


def garbage_triangle[
    T: DType
](mut A: Mat[T], keep_lower: Bool, diag: Bool = False):
    """The triangle that must not be read (and the diagonal) set to 1e3."""
    for i in range(A.rows):
        for j in range(A.cols):
            if (
                (j > i and keep_lower)
                or (j < i and not keep_lower)
                or (diag and i == j)
            ):
                A.set(i, j, 1e3)


def test_gemm[T: DType](mut ck: Check, mut r: Rng):
    var m = 7
    var n = 5
    var k = 6
    for ca in range(2):
        for cb in range(2):
            for cc in range(2):
                var A = Mat[T](m, k, ca == 1)
                var B = Mat[T](k, n, cb == 1)
                var C = Mat[T](m, n, cc == 1)
                A.fill(r)
                B.fill(r)
                C.fill(r)
                var C0 = C.copy()
                K.gemm(
                    m,
                    n,
                    k,
                    Float64(2.0),
                    False,
                    A.ptr(),
                    A.incRow(),
                    A.incCol(),
                    False,
                    B.ptr(),
                    B.incRow(),
                    B.incCol(),
                    Float64(0.5),
                    C.ptr(),
                    C.incRow(),
                    C.incCol(),
                )
                var e = 0.0
                for i in range(m):
                    for j in range(n):
                        var s = 0.0
                        for l in range(k):
                            s += A.get(i, l) * B.get(l, j)
                        e = max(
                            e, abs(C.get(i, j) - 2.0 * s - 0.5 * C0.get(i, j))
                        )
                ck.near("gemm " + ori(ca == 1) + ori(cb == 1) + ori(cc == 1), e)
    # gemmtr: only the triangle changes
    for lower in range(2):
        for cc in range(2):
            var nn = 200
            var A = Mat[T](nn, k, False)
            var B = Mat[T](k, nn, True)
            var C = Mat[T](nn, nn, cc == 1)
            A.fill(r)
            B.fill(r)
            C.fill(r)
            var C0 = C.copy()
            K.gemmtr(
                lower == 1,
                nn,
                k,
                Float64(1.0),
                False,
                A.ptr(),
                A.incRow(),
                A.incCol(),
                False,
                B.ptr(),
                B.incRow(),
                B.incCol(),
                Float64(1.0),
                C.ptr(),
                C.incRow(),
                C.incCol(),
            )
            var e = 0.0
            for i in range(nn):
                for j in range(nn):
                    var inside = (j <= i) if lower == 1 else (j >= i)
                    if inside:
                        var s = 0.0
                        for l in range(k):
                            s += A.get(i, l) * B.get(l, j)
                        e = max(e, abs(C.get(i, j) - s - C0.get(i, j)))
            ck.near("gemmtr " + String(lower) + ori(cc == 1), e)


def test_level2[T: DType](mut ck: Check, mut r: Rng):
    var m = 6
    var n = 4
    for ca in range(2):
        var A = Mat[T](m, n, ca == 1)
        A.fill(r)
        var x = Mat[T](n, 1, True)
        var y = Mat[T](m, 1, True)
        x.fill(r)
        y.fill(r)
        var y0 = y.copy()
        K.gemv(
            m,
            n,
            Float64(1.5),
            A.ptr(),
            A.incRow(),
            A.incCol(),
            x.ptr(),
            1,
            Float64(-1.0),
            y.ptr(),
            1,
        )
        var e = 0.0
        for i in range(m):
            var s = 0.0
            for j in range(n):
                s += A.get(i, j) * x.get(j, 0)
            e = max(e, abs(y.get(i, 0) - 1.5 * s + y0.get(i, 0)))
        ck.near("gemv " + ori(ca == 1), e)
        # sylmv: the lower triangle of a symmetric matrix
        var S = spd[T](m, ca == 1, r)
        var Sg = S.copy()
        garbage_triangle(Sg, True)
        var u = Mat[T](m, 1, True)
        var v = Mat[T](m, 1, True)
        u.fill(r)
        K.sylmv(
            m,
            Float64(1.0),
            Sg.ptr(),
            Sg.incRow(),
            Sg.incCol(),
            u.ptr(),
            1,
            Float64(0.0),
            v.ptr(),
            1,
        )
        e = 0.0
        for i in range(m):
            var s = 0.0
            for j in range(m):
                s += S.get(i, j) * u.get(j, 0)
            e = max(e, abs(v.get(i, 0) - s))
        ck.near("sylmv " + ori(ca == 1), e)
        _ = Sg^


def test_sym[T: DType](mut ck: Check, mut r: Rng):
    var m = 5
    var n = 3
    for lower in range(2):
        for ca in range(2):
            for cb in range(2):
                for cc in range(2):
                    var S = spd[T](m, ca == 1, r)
                    var Sg = S.copy()
                    garbage_triangle(Sg, lower == 1)
                    var B = Mat[T](m, n, cb == 1)
                    var C = Mat[T](m, n, cc == 1)
                    B.fill(r)
                    if lower == 1:
                        K.sylmm(
                            m,
                            n,
                            Float64(1.0),
                            Sg.ptr(),
                            Sg.incRow(),
                            Sg.incCol(),
                            B.ptr(),
                            B.incRow(),
                            B.incCol(),
                            Float64(0.0),
                            C.ptr(),
                            C.incRow(),
                            C.incCol(),
                        )
                    else:
                        K.syumm(
                            m,
                            n,
                            Float64(1.0),
                            Sg.ptr(),
                            Sg.incRow(),
                            Sg.incCol(),
                            B.ptr(),
                            B.incRow(),
                            B.incCol(),
                            Float64(0.0),
                            C.ptr(),
                            C.incRow(),
                            C.incCol(),
                        )
                    var e = 0.0
                    for i in range(m):
                        for j in range(n):
                            var s = 0.0
                            for l in range(m):
                                s += S.get(i, l) * B.get(l, j)
                            e = max(e, abs(C.get(i, j) - s))
                    ck.near(
                        "symm "
                        + String(lower)
                        + ori(ca == 1)
                        + ori(cb == 1)
                        + ori(cc == 1),
                        e,
                    )
                    _ = Sg^
        # syrk
        for ca in range(2):
            for cc in range(2):
                var A = Mat[T](m, n, ca == 1)
                A.fill(r)
                var C = Mat[T](m, m, cc == 1)
                garbage_triangle(C, lower == 1)
                var C0 = C.copy()
                if lower == 1:
                    K.sylrk(
                        m,
                        n,
                        Float64(1.0),
                        A.ptr(),
                        A.incRow(),
                        A.incCol(),
                        Float64(0.0),
                        C.ptr(),
                        C.incRow(),
                        C.incCol(),
                    )
                else:
                    K.syurk(
                        m,
                        n,
                        Float64(1.0),
                        A.ptr(),
                        A.incRow(),
                        A.incCol(),
                        Float64(0.0),
                        C.ptr(),
                        C.incRow(),
                        C.incCol(),
                    )
                var e = 0.0
                for i in range(m):
                    for j in range(m):
                        var inside = (j <= i) if lower == 1 else (j >= i)
                        var s = 0.0
                        for l in range(n):
                            s += A.get(i, l) * A.get(j, l)
                        e = max(
                            e,
                            abs(C.get(i, j) - (s if inside else C0.get(i, j))),
                        )
                ck.near(
                    "syrk " + String(lower) + ori(ca == 1) + ori(cc == 1), e
                )


def test_tri[T: DType](mut ck: Check, mut r: Rng):
    var m = 6
    var n = 4
    for lower in range(2):
        for unit in range(2):
            for ca in range(2):
                # a well-conditioned triangle: diagonal 4
                var A = Mat[T](m, m, ca == 1)
                A.fill(r)
                for i in range(m):
                    A.set(i, i, 4.0)
                var Ag = A.copy()
                garbage_triangle(Ag, lower == 1, unit == 1)

                def tri(
                    A: Mat[T], i: Int, j: Int, lower: Bool, unit: Bool
                ) -> Float64:
                    if i == j:
                        return 1.0 if unit else A.get(i, j)
                    if (j < i) == lower:
                        return A.get(i, j)
                    return 0.0

                for cb in range(2):
                    var B = Mat[T](m, n, cb == 1)
                    B.fill(r)
                    var X = B.copy()
                    # trmm: X := 2 A B
                    if lower == 1:
                        K.trlmm(
                            m,
                            n,
                            Float64(2.0),
                            False,
                            unit == 1,
                            Ag.ptr(),
                            Ag.incRow(),
                            Ag.incCol(),
                            X.ptr(),
                            X.incRow(),
                            X.incCol(),
                        )
                    else:
                        K.trumm(
                            m,
                            n,
                            Float64(2.0),
                            False,
                            unit == 1,
                            Ag.ptr(),
                            Ag.incRow(),
                            Ag.incCol(),
                            X.ptr(),
                            X.incRow(),
                            X.incCol(),
                        )
                    var e = 0.0
                    for i in range(m):
                        for j in range(n):
                            var s = 0.0
                            for l in range(m):
                                s += tri(
                                    A, i, l, lower == 1, unit == 1
                                ) * B.get(l, j)
                            e = max(e, abs(X.get(i, j) - 2.0 * s))
                    var tag = (
                        String(lower)
                        + String(unit)
                        + ori(ca == 1)
                        + ori(cb == 1)
                    )
                    ck.near("trmm " + tag, e)
                    # trsm: A X = 2 B
                    var Y = B.copy()
                    if lower == 1:
                        K.trlsm(
                            m,
                            n,
                            Float64(2.0),
                            False,
                            unit == 1,
                            Ag.ptr(),
                            Ag.incRow(),
                            Ag.incCol(),
                            Y.ptr(),
                            Y.incRow(),
                            Y.incCol(),
                        )
                    else:
                        K.trusm(
                            m,
                            n,
                            Float64(2.0),
                            False,
                            unit == 1,
                            Ag.ptr(),
                            Ag.incRow(),
                            Ag.incCol(),
                            Y.ptr(),
                            Y.incRow(),
                            Y.incCol(),
                        )
                    e = 0.0
                    for i in range(m):
                        for j in range(n):
                            var s = 0.0
                            for l in range(m):
                                s += tri(
                                    A, i, l, lower == 1, unit == 1
                                ) * Y.get(l, j)
                            e = max(e, abs(s - 2.0 * B.get(i, j)))
                    ck.near("trsm " + tag, e)
                    # trtrs with trans: A^T X = B
                    var Z = B.copy()
                    _ = K.trtrs(
                        lower == 1,
                        True,
                        False,
                        unit == 1,
                        m,
                        n,
                        Ag.ptr(),
                        Ag.incRow(),
                        Ag.incCol(),
                        Z.ptr(),
                        Z.incRow(),
                        Z.incCol(),
                    )
                    e = 0.0
                    for i in range(m):
                        for j in range(n):
                            var s = 0.0
                            for l in range(m):
                                s += tri(
                                    A, l, i, lower == 1, unit == 1
                                ) * Z.get(l, j)
                            e = max(e, abs(s - B.get(i, j)))
                    ck.near("trtrs^T " + tag, e)
                # trsv
                var x = Mat[T](m, 1, True)
                x.fill(r)
                var x0 = x.copy()
                if lower == 1:
                    K.trlsv(
                        m,
                        unit == 1,
                        Ag.ptr(),
                        Ag.incRow(),
                        Ag.incCol(),
                        x.ptr(),
                        1,
                    )
                else:
                    K.trusv(
                        m,
                        unit == 1,
                        Ag.ptr(),
                        Ag.incRow(),
                        Ag.incCol(),
                        x.ptr(),
                        1,
                    )
                var e = 0.0
                for i in range(m):
                    var s = 0.0
                    for l in range(m):
                        s += tri(A, i, l, lower == 1, unit == 1) * x.get(l, 0)
                    e = max(e, abs(s - x0.get(i, 0)))
                ck.near(
                    "trsv " + String(lower) + String(unit) + ori(ca == 1), e
                )
                _ = Ag^


def inverse_error[T: DType](A: Mat[T], Ai: Mat[T], lower: Int) -> Float64:
    """Max |A^-1 A - I|; lower: Ai holds only that triangle of a symmetric
    inverse (1: lower, 0: upper, -1: all of it)."""
    var n = A.rows
    var e = 0.0
    for i in range(n):
        for j in range(n):
            var s = 0.0
            for l in range(n):
                var a = Ai.get(i, l)
                if lower == 1 and l > i:
                    a = Ai.get(l, i)
                if lower == 0 and l < i:
                    a = Ai.get(l, i)
                s += a * A.get(l, j)
            e = max(e, abs(s - (1.0 if i == j else 0.0)))
    return e


def test_lapack[T: DType](mut ck: Check, mut r: Rng):
    var n = 6
    for ca in range(2):
        # potrf, potri
        for lower in range(2):
            var A = spd[T](n, ca == 1, r)
            var F = A.copy()
            garbage_triangle(F, lower == 1)
            var info = K.potrf(lower == 1, n, F.ptr(), F.incRow(), F.incCol())
            var e = 0.0
            for i in range(n):
                for j in range(n):
                    var s = 0.0
                    for l in range(n):
                        if lower == 1:  # L L^T
                            if l <= i and l <= j:
                                s += F.get(i, l) * F.get(j, l)
                        elif l <= i and l <= j:  # U^T U
                            s += F.get(l, i) * F.get(l, j)
                    e = max(e, abs(s - A.get(i, j)))
            ck.near(
                "potrf "
                + String(lower)
                + ori(ca == 1)
                + " info "
                + String(info),
                e + Float64(abs(info)),
            )
            info = K.potri(lower == 1, n, F.ptr(), F.incRow(), F.incCol())
            ck.near(
                "potri " + String(lower) + ori(ca == 1),
                inverse_error(A, F, lower) + Float64(abs(info)),
            )
        # getrf, getrs (both transposes, B in both orientations), getri
        var A = Mat[T](n, n, ca == 1)
        A.fill(r)
        var LU = A.copy()
        var piv = List[Int32](length=n, fill=0)
        var pp = piv.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var info = K.getrf(n, n, LU.ptr(), LU.incRow(), LU.incCol(), pp, 1)
        ck.near("getrf info", Float64(abs(info)))
        for tr in range(2):
            for cb in range(2):
                var B = Mat[T](n, 3, cb == 1)
                B.fill(r)
                var X = B.copy()
                K.getrs(
                    tr == 1,
                    False,
                    n,
                    3,
                    LU.ptr(),
                    LU.incRow(),
                    LU.incCol(),
                    pp,
                    1,
                    X.ptr(),
                    X.incRow(),
                    X.incCol(),
                )
                var e = 0.0
                for i in range(n):
                    for j in range(3):
                        var s = 0.0
                        for l in range(n):
                            s += (
                                A.get(l, i) if tr == 1 else A.get(i, l)
                            ) * X.get(l, j)
                        e = max(e, abs(s - B.get(i, j)))
                ck.near("getrs " + String(tr) + ori(ca == 1) + ori(cb == 1), e)
        info = K.getri(n, LU.ptr(), LU.incRow(), LU.incCol(), pp, 1)
        ck.near(
            "getri " + ori(ca == 1),
            inverse_error(A, LU, -1) + Float64(abs(info)),
        )
        # getrf with 64-bit pivots
        var LU2 = A.copy()
        var piv64 = List[Int64](length=n, fill=0)
        var p64 = piv64.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        _ = K.getrf(n, n, LU2.ptr(), LU2.incRow(), LU2.incCol(), p64, 1)
        info = K.getri(n, LU2.ptr(), LU2.incRow(), LU2.incCol(), p64, 1)
        ck.near(
            "getri int64 pivots " + ori(ca == 1),
            inverse_error(A, LU2, -1) + Float64(abs(info)),
        )
        _ = piv^
        _ = piv64^
        # sytrf, sytrs: symmetric indefinite
        var S = Mat[T](n, n, ca == 1)
        for i in range(n):
            for j in range(i + 1):
                var v = r.next()
                S.set(i, j, v)
                S.set(j, i, v)
        var D = S.copy()
        var spiv = List[Int32](length=n, fill=0)
        var sp = spiv.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        info = K.sytrf(n, n, D.ptr(), D.incRow(), D.incCol(), sp, 1)
        for cb in range(2):
            var B = Mat[T](n, 2, cb == 1)
            B.fill(r)
            var X = B.copy()
            K.sytrs(
                True,
                False,
                n,
                2,
                D.ptr(),
                D.incRow(),
                D.incCol(),
                sp,
                1,
                X.ptr(),
                X.incRow(),
                X.incCol(),
            )
            var e = 0.0
            for i in range(n):
                for j in range(2):
                    var s = 0.0
                    for l in range(n):
                        s += S.get(i, l) * X.get(l, j)
                    e = max(e, abs(s - B.get(i, j)))
            ck.near(
                "sytrs " + ori(ca == 1) + ori(cb == 1), e + Float64(abs(info))
            )
        _ = D^
        _ = spiv^
        # gels: least squares, A^T (A x - b) = 0
        var m = 9
        for cb in range(2):
            var G = Mat[T](m, n, ca == 1)
            G.fill(r)
            var B = Mat[T](m, 2, cb == 1)
            B.fill(r)
            var G0 = G.copy()
            var X = B.copy()
            info = K.gels(
                m,
                n,
                2,
                G.ptr(),
                G.incRow(),
                G.incCol(),
                X.ptr(),
                X.incRow(),
                X.incCol(),
            )
            var e = 0.0
            for j in range(n):
                for c in range(2):
                    var s = 0.0
                    for i in range(m):
                        var res = -B.get(i, c)
                        for l in range(n):
                            res += G0.get(i, l) * X.get(l, c)
                        s += G0.get(i, j) * res
                    e = max(e, abs(s))
            ck.near(
                "gels " + ori(ca == 1) + ori(cb == 1), e + Float64(abs(info))
            )
            _ = G^


def test_helpers[T: DType](mut ck: Check, mut r: Rng):
    var m = 70
    var n = 66
    for cx in range(2):
        for cy in range(2):
            var X = Mat[T](m, n, cx == 1)
            var Y = Mat[T](m, n, cy == 1)
            X.fill(r)
            Y.fill(r)
            var Y0 = Y.copy()
            K.geaxpy(
                m,
                n,
                Float64(-2.0),
                X.ptr(),
                X.incRow(),
                X.incCol(),
                Y.ptr(),
                Y.incRow(),
                Y.incCol(),
            )
            var e = 0.0
            for i in range(m):
                for j in range(n):
                    e = max(
                        e, abs(Y.get(i, j) - Y0.get(i, j) + 2.0 * X.get(i, j))
                    )
            ck.near("geaxpy " + ori(cx == 1) + ori(cy == 1), e)
            var Z = Mat[T](m, n, cy == 1)
            K.gecopy(
                m,
                n,
                X.ptr(),
                X.incRow(),
                X.incCol(),
                Z.ptr(),
                Z.incRow(),
                Z.incCol(),
            )
            e = 0.0
            for i in range(m):
                for j in range(n):
                    e = max(e, abs(Z.get(i, j) - X.get(i, j)))
            ck.near("gecopy " + ori(cx == 1) + ori(cy == 1), e)
    var A = Mat[T](n, n, True)
    A.fill(r)
    var A0 = A.copy()
    K.gescal(n, n, Float64(3.0), A.ptr(), A.incRow(), A.incCol())
    K.adddiag(n, Float64(1.0), A.ptr(), A.incRow(), A.incCol())
    var e = 0.0
    for i in range(n):
        for j in range(n):
            e = max(
                e,
                abs(
                    A.get(i, j) - 3.0 * A0.get(i, j) - (1.0 if i == j else 0.0)
                ),
            )
    ck.near("gescal adddiag", e)
    for lower in range(2):
        var S = Mat[T](n, n, False)
        S.fill(r)
        var S0 = S.copy()
        K.sycomplete(lower == 1, n, S.ptr(), S.incRow(), S.incCol())
        e = 0.0
        for i in range(n):
            for j in range(n):
                var keep = (j <= i) if lower == 1 else (j >= i)
                var v = S0.get(i, j) if keep else S0.get(j, i)
                e = max(e, abs(S.get(i, j) - v))
        ck.near("sycomplete " + String(lower), e)


def run[T: DType](tol: Float64) -> Int:
    var ck = Check(tol)
    var r = Rng(42)
    test_gemm[T](ck, r)
    test_level2[T](ck, r)
    test_sym[T](ck, r)
    test_tri[T](ck, r)
    test_lapack[T](ck, r)
    test_helpers[T](ck, r)
    print(String(T), ":", ck.failed, "failures")
    return ck.failed


def main():
    var failed = run[DType.float64](1e-11) + run[DType.float32](2e-4)
    print("PASS" if failed == 0 else "FAIL")
