"""lam.frontend: write linear algebra in Python, get ulm_lam IR.

    from ulm import lam
    from ulm.lam import Matrix, Vector

    @lam.function
    def gls(X: Matrix(2000, 200, "full_rank"), M: Matrix(2000, 2000, "spd"),
            y: Vector(2000)):
        return (X.T @ M.inv @ X).inv @ X.T @ M.inv @ y

Calling a function with ulm_lam values traces it: operators build ulm_lam
operations (@ is ulm_lam.mul, + and - are ulm_lam.add, scalar * is ulm_lam.scale,
.T is ulm_lam.transpose, .inv is ulm_lam.inverse).  Shapes are checked while
tracing, so a mismatch is a Python error at the line that causes it.
Matrix properties are declared on parameters; ulm-lam-infer-properties derives
those of intermediate results.

Parameter intents (the ulm.lam ABI): matrices are passed by reference.
    Matrix(...)            in: read only (the default)
    InOut(Matrix(...))     the function may overwrite it; with
                           lam.update(param, value) it holds `value` on
                           exit, otherwise its contents are unspecified
    return values          out: buffers the caller provides

Sizes are numbers, or names that are run-time parameters:
    Matrix("n", "m", "full_rank")   n x m, n and m given by the arrays
The generated C function takes them as size_t parameters in front of the
matrices (in order of first appearance); from Python they are taken from
the arrays.  Equal names are equal sizes.

Loops and external functions:
    observe = lam.extern("observe", [Index], Vector(2000))
    with lam.loop(n) as L:
        y = observe(L.index)
        ...
    with lam.loop(n, carry=[X]) as L:      # loop-carried matrices
        (X,) = L.carried
        L.next(X + A @ B @ C)
    (X,) = L.results

The IR is plain text in the syntax of the ulm_lam dialect; no MLIR Python
bindings are needed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

PROPERTIES = (
    "square", "symmetric", "spsd", "spd", "lower_tri", "upper_tri",
    "unit_diag", "identity", "zero", "orthogonal", "permutation",
    "non_singular", "full_rank", "diagonal",
)


class LamError(Exception):
    """An error in a ulm_lam program (shapes, scopes, signatures)."""


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatrixType:
    rows: int | str  # a number, or the name of a run-time size
    cols: int | str
    props: frozenset = frozenset()
    intent: str = "in"  # in | inout (parameters only)

    def mlir(self, with_props: bool = True) -> str:
        def d(x):
            return "?" if isinstance(x, str) else str(x)
        t = f"tensor<{d(self.rows)}x{d(self.cols)}xf64"
        if with_props and self.props:
            t += ", #ulm_lam.props<" + ", ".join(sorted(self.props)) + ">"
        return t + ">"

    def plain(self) -> "MatrixType":
        return MatrixType(self.rows, self.cols)

    def symbols(self) -> list[str]:
        """The names of its run-time sizes."""
        return [d for d in (self.rows, self.cols) if isinstance(d, str)]

    def dims_attr(self, position: dict) -> str:
        """`ulm_lam.dims = [...]`: the argument positions of its run-time
        sizes ("" if it has none)."""
        syms = self.symbols()
        if not syms:
            return ""
        return "ulm_lam.dims = [" + ", ".join(str(position[x])
                                              for x in syms) + "]"

    def __str__(self) -> str:
        p = (" " + ",".join(sorted(self.props))) if self.props else ""
        return f"{self.rows}x{self.cols}{p}"


@dataclass(frozen=True)
class IndexType:
    def mlir(self, with_props: bool = True) -> str:
        return "index"

    def __str__(self) -> str:
        return "index"


@dataclass(frozen=True)
class FloatType:
    def mlir(self, with_props: bool = True) -> str:
        return "f64"

    def __str__(self) -> str:
        return "f64"


Index = IndexType()
Float = FloatType()


def _symbols(types) -> list[str]:
    """The run-time sizes of these types, in order of first appearance."""
    out = []
    for t in types:
        if isinstance(t, MatrixType):
            for x in t.symbols():
                if x not in out:
                    out.append(x)
    return out


def Matrix(rows: int | str, cols: int | str, *props: str,
           stored: str | None = None) -> MatrixType:
    """An m x n matrix of doubles with properties, e.g. Matrix(n, n, "spd").

    stored="lower" or "upper" (symmetric matrices only): only this triangle
    is referenced, the other one may hold anything (LAPACK's uplo).  By
    default all elements are stored and read."""
    for p in props:
        if p not in PROPERTIES:
            raise LamError(f"unknown matrix property {p!r}; known: "
                            + ", ".join(PROPERTIES))
    for d in (rows, cols):
        if isinstance(d, str):
            if not d.isidentifier():
                raise LamError(f"a run-time size must be a name, got {d!r}")
        elif not isinstance(d, int) or d <= 0:
            raise LamError(f"matrix sizes must be positive numbers or "
                            f"names, got {rows}x{cols}")
    ps = set(props)
    if stored is not None:
        if stored not in ("lower", "upper"):
            raise LamError(f"stored must be 'lower' or 'upper', got "
                            f"{stored!r}")
        if not ps & {"symmetric", "spsd", "spd", "identity"}:
            raise LamError("stored= needs a symmetric matrix (symmetric, "
                            "spsd or spd)")
        ps.add(f"{stored}_stored")
    return MatrixType(rows, cols, frozenset(ps))


def Vector(n: int | str, *props: str) -> MatrixType:
    """An n x 1 column vector."""
    return Matrix(n, 1, *props)


def InOut(t: MatrixType) -> MatrixType:
    """A matrix parameter that the function may overwrite."""
    if not isinstance(t, MatrixType):
        raise LamError("InOut needs a matrix type")
    return MatrixType(t.rows, t.cols, t.props, "inout")


def In(t: MatrixType) -> MatrixType:
    """A read-only matrix parameter (the default)."""
    return MatrixType(t.rows, t.cols, t.props, "in")


# ---------------------------------------------------------------------------
# Builder: textual IR with nested regions
# ---------------------------------------------------------------------------


class _Region:
    _ids = 0

    def __init__(self, indent: str):
        _Region._ids += 1
        self.id = _Region._ids
        self.lines: list[str] = []
        self.indent = indent
        self.constants: dict[tuple, "Value"] = {}


class _Builder:
    current: "_Builder | None" = None

    def __init__(self):
        self.counter = 0
        self.regions = [_Region("  ")]

    def fresh(self) -> str:
        self.counter += 1
        return f"%v{self.counter}"

    @property
    def region(self) -> _Region:
        return self.regions[-1]

    def emit(self, line: str) -> None:
        self.region.lines.append(self.region.indent + line)

    def live(self, region_id: int) -> bool:
        return any(r.id == region_id for r in self.regions)

    def constant(self, value, t) -> "Value":
        key = (repr(value), str(t))
        # constants of an enclosing region are visible
        for r in reversed(self.regions):
            if key in r.constants:
                return r.constants[key]
        v = Value(self.fresh(), t, self.region.id)
        if isinstance(t, IndexType):
            self.emit(f"{v.name} = arith.constant {int(value)} : index")
        else:
            self.emit(f"{v.name} = arith.constant {float(value):.17e} : f64")
        self.region.constants[key] = v
        return v


def _builder() -> _Builder:
    if _Builder.current is None:
        raise LamError("ulm_lam values exist only while a ulm.lam function is "
                        "traced")
    return _Builder.current


# ---------------------------------------------------------------------------
# Values and operations
# ---------------------------------------------------------------------------


class Value:
    """A value of a traced ulm.lam function (matrix, index or float)."""

    def __init__(self, name: str, type_, region_id: int):
        self.name = name
        self.type = type_
        self.region_id = region_id
        self._t = None
        self._inv = None

    def _use(self) -> str:
        if not _builder().live(self.region_id):
            raise LamError(f"{self.name} ({self.type}) is used outside the "
                            "loop that defines it")
        return self.name

    def _matrix(self, what: str) -> MatrixType:
        if not isinstance(self.type, MatrixType):
            raise LamError(f"{what} needs a matrix, got {self.type}")
        return self.type

    # -- algebra ----------------------------------------------------------

    @property
    def T(self) -> "Value":
        t = self._matrix("transpose")
        if self._t is None or not _builder().live(self._t.region_id):
            r = MatrixType(t.cols, t.rows)
            self._t = _op("ulm_lam.transpose", [self], r)
        return self._t

    @property
    def inv(self) -> "Value":
        t = self._matrix("inverse")
        if t.rows != t.cols:
            raise LamError(f"inverse of a non-square {t} matrix")
        if self._inv is None or not _builder().live(self._inv.region_id):
            self._inv = _op("ulm_lam.inverse", [self], MatrixType(t.rows, t.cols))
        return self._inv

    def __matmul__(self, other: "Value") -> "Value":
        a = self._matrix("@")
        b = _as_value(other)._matrix("@")
        if a.cols != b.rows:
            raise LamError(f"product of {a} and {b}: {a.cols} columns but "
                            f"{b.rows} rows")
        return _op("ulm_lam.mul", [self, other], MatrixType(a.rows, b.cols))

    def __add__(self, other) -> "Value":
        if isinstance(self.type, FloatType):
            return _scalar_op("addf", self, other)
        other = _as_value(other)
        a, b = self._matrix("+"), other._matrix("+")
        if (a.rows, a.cols) != (b.rows, b.cols):
            raise LamError(f"sum of {a} and {b}: different shapes")
        return _op("ulm_lam.add", [self, other], MatrixType(a.rows, a.cols))

    def __sub__(self, other) -> "Value":
        if isinstance(self.type, FloatType):
            return _scalar_op("subf", self, other)
        return self + (-1.0) * _as_value(other)

    def __neg__(self) -> "Value":
        if isinstance(self.type, FloatType):
            return _scalar_op("subf", 0.0, self)
        return (-1.0) * self

    # scalar arithmetic of Float values (1 / beta, alpha / beta, ...)
    def __radd__(self, other) -> "Value":
        return _scalar_op("addf", other, self)

    def __rsub__(self, other) -> "Value":
        return _scalar_op("subf", other, self)

    def __truediv__(self, other) -> "Value":
        if isinstance(self.type, MatrixType):  # A / s
            if isinstance(other, (int, float)):
                return _scale(1.0 / other, self)
            return _scale_by(_scalar_op("divf", 1.0, other), self)
        return _scalar_op("divf", self, other)

    def __rtruediv__(self, other) -> "Value":
        return _scalar_op("divf", other, self)

    def __mul__(self, alpha) -> "Value":
        if isinstance(self.type, FloatType):  # s * M, s a Float value
            if isinstance(alpha, Value) and isinstance(alpha.type,
                                                       MatrixType):
                return _scale_by(self, alpha)
            return _scalar_op("mulf", self, alpha)
        if isinstance(alpha, Value) and isinstance(alpha.type, FloatType):
            return _scale_by(alpha, self)
        if isinstance(alpha, (int, float)):
            return _scale(float(alpha), self)
        if isinstance(alpha, Value) and isinstance(alpha.type, FloatType):
            return _scale_by(alpha, self)
        raise LamError("* is scaling by a number or a Float parameter; use "
                       "@ for products")

    __rmul__ = __mul__

    def __pow__(self, k: int) -> "Value":
        """A ** k: products (k > 0), the identity (k = 0), inverses (k < 0)."""
        a = self._matrix("**")
        if not isinstance(k, int):
            raise LamError("A ** k needs an integer k")
        if a.rows != a.cols:
            raise LamError(f"A ** k needs a square matrix, got {a}")
        if k == 0:
            return eye(a.rows)
        if k < 0:
            return inv(self) ** (-k)
        return self if k == 1 else self @ self ** (k - 1)

    def __repr__(self) -> str:
        return f"<ulm_lam value {self.name}: {self.type}>"


def _as_value(x) -> Value:
    if isinstance(x, Value):
        return x
    raise LamError(f"expected a ulm_lam value, got {type(x).__name__}")


def _op(name: str, operands: list[Value], result: MatrixType) -> Value:
    b = _builder()
    names = ", ".join(o._use() for o in operands)
    types = ", ".join(o.type.mlir() for o in operands)
    v = Value(b.fresh(), result, b.region.id)
    b.emit(f"{v.name} = {name} {names} : {types} -> {result.mlir()}")
    return v


def _scale(alpha: float, x: Value) -> Value:
    t = x._matrix("scaling")
    b = _builder()
    c = b.constant(alpha, Float)
    r = MatrixType(t.rows, t.cols)
    v = Value(b.fresh(), r, b.region.id)
    b.emit(f"{v.name} = ulm_lam.scale {c.name}, {x._use()} : f64, {t.mlir()} "
           f"-> {r.mlir()}")
    return v


def _scalar_op(op: str, x, y) -> Value:
    """x op y for Float values and numbers (arith.addf, subf, mulf, divf)."""
    b = _builder()
    vals = []
    for a in (x, y):
        if isinstance(a, (int, float)):
            a = b.constant(float(a), Float)
        if not (isinstance(a, Value) and isinstance(a.type, FloatType)):
            raise LamError(f"{op}: scalars only, got {a}")
        vals.append(a)
    v = Value(b.fresh(), Float, b.region.id)
    b.emit(f"{v.name} = arith.{op} {vals[0]._use()}, {vals[1]._use()} : f64")
    return v


def _scale_by(alpha: Value, x: Value) -> Value:
    """alpha * x for a scalar known at run time (a Float parameter)."""
    t = x._matrix("scaling")
    b = _builder()
    r = MatrixType(t.rows, t.cols)
    v = Value(b.fresh(), r, b.region.id)
    b.emit(f"{v.name} = ulm_lam.scale {alpha._use()}, {x._use()} : f64, "
           f"{t.mlir()} -> {r.mlir()}")
    return v


def eye(n: int | str) -> Value:
    """The n x n identity; n a number or the name of a run-time size of the
    function (e.g. K + s2 * lam.eye("n"): kernel selection adds s2 to the
    diagonal, the identity is never formed)."""
    t = Matrix(n, n, "identity")
    b = _builder()
    sizes = _size_values([n], "eye") * 2 if isinstance(n, str) else []
    v = Value(b.fresh(), t, b.region.id)
    operands = f"({', '.join(sizes)})" if sizes else ""
    b.emit(f"{v.name} = ulm_lam.identity{operands} : {t.mlir()}")
    return v


def assume(x: Value, *props: str) -> Value:
    """x with properties the compiler cannot derive, e.g.
    assume(K + s2 * eye("n"), "spd") for s2 > 0 and K spsd.  Trusted, not
    checked; a Cholesky factorisation of a matrix that is not spd returns
    info > 0 (lam.InfoError in Python)."""
    x = _as_value(x)
    t = x._matrix("assume")
    r = Matrix(t.rows, t.cols, *props)
    b = _builder()
    v = Value(b.fresh(), r, b.region.id)
    b.emit(f"{v.name} = ulm_lam.assume {x._use()} : {t.mlir()} -> "
           f"{r.mlir()}")
    return v


def _plain(x: Value) -> Value:
    """`x` with a type without properties (dropping them is sound)."""
    t = x.type
    if not isinstance(t, MatrixType) or not t.props:
        return x
    b = _builder()
    v = Value(b.fresh(), t.plain(), b.region.id)
    b.emit(f"{v.name} = ulm_lam.assume {x._use()} : {t.mlir()} -> "
           f"{t.plain().mlir()}")
    return v


def inv(x: Value) -> Value:
    """The inverse (never formed explicitly: kernel selection solves)."""
    return _as_value(x).inv


def lstsq(a: Value, b: Value) -> Value:
    """The X that minimises ||A X - B|| (A m x n with m >= n and full column
    rank): QR (LAPACK's gels) where the backend has LAPACK, otherwise the
    normal equations with a Cholesky factorisation."""
    ta, tb = _as_value(a)._matrix("lstsq"), _as_value(b)._matrix("lstsq")
    if ta.rows != tb.rows:
        raise LamError(f"lstsq: A has {ta.rows} rows, B has {tb.rows}")
    return _op("ulm_lam.lstsq", [_as_value(a), _as_value(b)],
               MatrixType(ta.cols, tb.cols))


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


class loop:
    """for i in range(n), with optional loop-carried matrices.

        with lam.loop(n, carry=[X0]) as L:
            i = L.index
            (X,) = L.carried
            L.next(X + ...)
        (X,) = L.results
    """

    def __init__(self, n, carry=()):
        self.n = n
        self.inits = [_as_value(c) for c in carry]
        for c in self.inits:
            c._matrix("a loop-carried value")
        self.results: tuple = ()
        self._next = None

    def __enter__(self) -> "loop":
        b = _builder()
        self._b = b
        c0 = b.constant(0, Index)
        c1 = b.constant(1, Index)
        if isinstance(self.n, int):
            ub = b.constant(self.n, Index)
        else:
            ub = _as_value(self.n)
            if not isinstance(ub.type, IndexType):
                raise LamError("the trip count of a loop is an index")
        # carried types are plain: properties do not survive iterations
        self._inits = [_plain(c) for c in self.inits]
        self._head = (c0, c1, ub)
        self._result = b.fresh()
        b.regions.append(_Region(b.region.indent + "  "))
        self.index = Value(b.fresh(), Index, b.region.id)
        self.carried = tuple(Value(b.fresh(), c.type.plain(), b.region.id)
                             for c in self._inits)
        return self

    def next(self, *values) -> None:
        """The carried values of the next iteration."""
        if len(values) != len(self.carried):
            raise LamError(f"next() needs {len(self.carried)} values")
        vals = []
        for v, c in zip(values, self.carried):
            v = _plain(_as_value(v))
            if v.type.mlir() != c.type.mlir():  # not the intent
                raise LamError(f"loop-carried value: {v.type} instead of "
                                f"{c.type}")
            vals.append(v)
        self._next = vals

    def __exit__(self, exc_type, exc, tb) -> bool:
        b = self._b
        if exc_type is not None:
            b.regions.pop()
            return False
        if self.carried and self._next is None:
            raise LamError("a loop with carried values must call next()")
        body = b.regions.pop()
        c0, c1, ub = self._head
        k = len(self._inits)
        head = (f"scf.for {self.index.name} = {c0.name} to {ub._use()} "
                f"step {c1.name}")
        if k:
            iters = ", ".join(f"{c.name} = {i._use()}"
                              for c, i in zip(self.carried, self._inits))
            types = ", ".join(i.type.mlir() for i in self._inits)
            res = self._result + (f":{k}" if k > 1 else "")
            b.emit(f"{res} = {head} iter_args({iters}) -> ({types}) {{")
        else:
            b.emit(f"{head} {{")
        b.region.lines.extend(body.lines)
        if k:
            names = ", ".join(v.name for v in self._next)
            types = ", ".join(v.type.mlir() for v in self._next)
            b.emit(f"  scf.yield {names} : {types}")
        b.emit("}")
        names = ([self._result] if k == 1
                 else [f"{self._result}#{j}" for j in range(k)])
        self.results = tuple(Value(n, i.type, b.region.id)
                             for n, i in zip(names, self._inits))
        return False


# ---------------------------------------------------------------------------
# External functions
# ---------------------------------------------------------------------------


@dataclass
class Extern:
    """An external C function the ulm_lam program calls.

    extern("observe", [Index], Vector(2000))       y = observe(i)
    extern("record", [Index, Vector(200)])          record(i, b)
    extern("update", [Index, Matrix(n, n)], Matrix(n, n), updates=1)
                                         R = update(i, R) updates R in place
    extern("post", [Index, Vector(m)], workspace=True)
                                         takes (work, workspace_info) too
    """

    name: str
    params: list
    result: object = None
    updates: int | None = None
    workspace: bool = False

    def __post_init__(self):
        if self.updates is not None:
            if self.result is None or self.updates != len(self.params) - 1:
                raise LamError(
                    f"{self.name}: `updates` must name the last parameter, "
                    "and the result is its new value")
            if self.params[self.updates].plain() != self.result.plain():
                raise LamError(f"{self.name}: the updated parameter and the "
                                "result must have the same shape")

    @property
    def sizes(self) -> list[str]:
        """Its run-time sizes: index parameters in front of the others."""
        return _symbols(list(self.params) + [self.result])

    def declaration(self) -> str:
        sizes = self.sizes
        pos = {x: k for k, x in enumerate(sizes)}
        params = ", ".join(["index"] * len(sizes) +
                           [p.mlir() for p in self.params])
        res = ""
        if self.result is not None:
            attrs = []
            if self.updates is not None:
                attrs.append(f"ulm_lam.inout = {self.updates + len(sizes)}"
                             " : i64")
            if self.result.dims_attr(pos):
                attrs.append(self.result.dims_attr(pos))
            r = self.result.mlir()
            if attrs:
                r += " {" + ", ".join(attrs) + "}"
            res = f" -> ({r})"
        attrs = " attributes {ulm_lam.workspace}" if self.workspace else ""
        return f"func.func private @{self.name}({params}){res}{attrs}"

    def __call__(self, *args):
        b = _builder()
        if len(args) != len(self.params):
            raise LamError(f"{self.name} takes {len(self.params)} arguments")
        vals = []
        for a, p in zip(args, self.params):
            if isinstance(a, int) and isinstance(p, IndexType):
                a = b.constant(a, Index)
            a = _as_value(a)
            if isinstance(p, MatrixType):
                if not isinstance(a.type, MatrixType) or (
                        a.type.rows, a.type.cols) != (p.rows, p.cols):
                    raise LamError(f"{self.name}: argument {a.type} instead "
                                    f"of {p}")
                if a.type.mlir() != p.mlir():
                    a = _cast(a, p)
            elif a.type != p:
                raise LamError(f"{self.name}: argument {a.type} instead of "
                                f"{p}")
            vals.append(a)
        _Program.current.externs[self.name] = self
        names = ", ".join(_size_values(self.sizes, self.name) +
                          [v._use() for v in vals])
        ptypes = ", ".join(["index"] * len(self.sizes) +
                           [p.mlir() for p in self.params])
        if self.result is None:
            b.emit(f"func.call @{self.name}({names}) : ({ptypes}) -> ()")
            return None
        v = Value(b.fresh(), self.result, b.region.id)
        b.emit(f"{v.name} = func.call @{self.name}({names}) : ({ptypes}) -> "
               f"{self.result.mlir()}")
        return v


def _size_values(sizes: list[str], callee: str) -> list[str]:
    """The values of run-time sizes in the function being traced (its
    size parameters, by name)."""
    f = Function.tracing
    known = f._size_values if f is not None else {}
    out = []
    for x in sizes:
        if x not in known:
            raise LamError(f"{callee} needs the size {x!r}, which no "
                            "parameter of the calling function has")
        out.append(known[x])
    return out


def extern(name, params, result=None, updates=None, workspace=False):
    return Extern(name, list(params), result, updates, workspace)


def _cast(x: Value, t: MatrixType) -> Value:
    """Change the asserted properties of `x` to those of `t` (ulm_lam.assume:
    dropping is sound; adding is an assertion of the program)."""
    b = _builder()
    v = Value(b.fresh(), MatrixType(t.rows, t.cols, t.props), b.region.id)
    b.emit(f"{v.name} = ulm_lam.assume {x._use()} : {x.type.mlir()} -> "
           f"{v.type.mlir()}")
    return v


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


class _Program:
    """The functions and externs a module needs (while tracing)."""

    current: "_Program | None" = None

    def __init__(self):
        self.externs: dict[str, Extern] = {}
        self.functions: list["Function"] = []


def update(param: Value, value: Value) -> None:
    """On exit, the InOut parameter `param` holds `value`."""
    f = Function.tracing
    if f is None:
        raise LamError("lam.update outside a ulm.lam function")
    f._update(param, value)


class Function:
    """A ulm.lam function (from @lam.function)."""

    tracing: "Function | None" = None

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        sig = inspect.signature(fn)
        self.params = []
        for p in sig.parameters.values():
            t = p.annotation
            if t is inspect.Parameter.empty:
                raise LamError(f"{self.name}: parameter {p.name} needs a "
                                "type (Matrix(...), Vector(...), Index)")
            if isinstance(t, str):
                t = eval(t, fn.__globals__)  # from __future__ annotations
            if not isinstance(t, (MatrixType, IndexType, FloatType)):
                raise LamError(f"{self.name}: parameter {p.name}: {t!r} is "
                                "not a ulm_lam type")
            self.params.append((p.name, t))
        # run-time sizes: index parameters in front of the others
        self.sizes = _symbols(t for _, t in self.params)
        self._size_values = {x: f"%s.{x}" for x in self.sizes}
        self._text = None
        self.results: list[MatrixType] = []

    # -- tracing ------------------------------------------------------------

    def trace(self) -> str:
        if self._text is not None:
            return self._text
        outer_b, outer_f = _Builder.current, Function.tracing
        b = _Builder()
        _Builder.current, Function.tracing = b, self
        self._updates: dict[int, Value] = {}
        try:
            self._args = [Value(f"%p.{n}", t, b.region.id)
                          for n, t in self.params]
            out = self.fn(*self._args)
            if out is None:
                out = ()
            elif not isinstance(out, tuple):
                out = (out,)
            outs = [_plain(_as_value(o)) for o in out]
            for o in outs:
                o._matrix("a result")
            def kept(k, v):
                # (the new value of a parameter stored as one triangle keeps
                # the storage, see _update; other properties are dropped)
                st = v.type.props & {"lower_stored", "upper_stored"}
                if not st or not (self.params[k][1].props & st):
                    return _plain(v)
                if v.type.props == st | {"symmetric"}:
                    return v
                return _cast(v, MatrixType(v.type.rows, v.type.cols,
                                           frozenset(st | {"symmetric"})))
            ups = [(k, kept(k, v)) for k, v in sorted(self._updates.items())]
            results = outs + [v for _, v in ups]
            pos = {x: k for k, x in enumerate(self.sizes)}
            for v in outs:
                for x in v.type.symbols():
                    if x not in pos:
                        raise LamError(f"{self.name}: the size {x!r} of a "
                                        "result is not a size of a parameter")

            def attrs(t, a):
                d = t.dims_attr(pos)
                return "{" + a + (", " + d if d else "") + "}"
            nsz = len(self.sizes)
            rtypes = [f"{v.type.mlir()} {attrs(v.type, 'ulm_lam.out')}"
                      for v in outs]
            rtypes += [f"{v.type.mlir()} "
                       f"{attrs(v.type, f'ulm_lam.inout = {k + nsz} : i64')}"
                       for k, v in ups]
            self.results = [v.type for v in outs]
            self.updated = [k for k, _ in ups]
            # the types of the new values of the updated parameters (after
            # the results: a call returns them so)
            self.update_types = [v.type for _, v in ups]
            ret = ", ".join(v._use() for v in results)
            rett = ", ".join(v.type.mlir() for v in results)
            b.emit(f"return {ret} : {rett}" if results else "return")
        finally:
            _Builder.current, Function.tracing = outer_b, outer_f
        pos = {x: k for k, x in enumerate(self.sizes)}
        args = [f"%s.{x}: index" for x in self.sizes]
        for n, t in self.params:
            attr = ""
            if isinstance(t, MatrixType):
                a = ["ulm_lam.inout" if t.intent == "inout" else "ulm_lam.in"]
                if t.dims_attr(pos):
                    a.append(t.dims_attr(pos))
                attr = " {" + ", ".join(a) + "}"
            args.append(f"%p.{n}: {t.mlir()}{attr}")
        res = f" -> ({', '.join(rtypes)})" if rtypes else ""
        head = f"func.func @{self.name}({', '.join(args)}){res} {{"
        self._text = "\n".join([head] + b.region.lines + ["}"])
        return self._text

    def _update(self, param: Value, value: Value) -> None:
        for k, a in enumerate(self._args):
            if a is param:
                t = self.params[k][1]
                if not isinstance(t, MatrixType) or t.intent != "inout":
                    raise LamError(f"lam.update: parameter {self.params[k][0]}"
                                    " must be InOut(...)")
                v = _as_value(value)
                if (v.type.rows, v.type.cols) != (t.rows, t.cols):
                    raise LamError(f"lam.update: {v.type} instead of {t}")
                # a parameter stored as one triangle keeps its storage: the
                # new value needs only that triangle (LAPACK's uplo), no
                # completion of the other one
                stored = t.props & {"lower_stored", "upper_stored"}
                if stored and isinstance(v, Value):
                    sym = t.props & {"symmetric", "spsd", "spd"}
                    v = _cast(_plain(v), MatrixType(
                        v.type.rows, v.type.cols,
                        frozenset(v.type.props | sym | stored)))
                self._updates[k] = v
                return
        raise LamError("lam.update: the first argument must be a parameter "
                        "of the function")

    # -- calls --------------------------------------------------------------

    def __call__(self, *args):
        if _Builder.current is None:
            raise LamError(f"{self.name} is a ulm.lam function: compile it "
                            "with lam.compile, or call it from another ulm.lam "
                            "function")
        return self._call(*args)

    def _call(self, *args):
        """A call from another ulm.lam function."""
        self.trace()
        b = _builder()
        prog = _Program.current
        if self not in prog.functions:
            prog.functions.append(self)
        vals = []
        for a, (n, t) in zip(args, self.params):
            if isinstance(a, int) and isinstance(t, IndexType):
                a = b.constant(a, Index)
            a = _as_value(a)
            if isinstance(t, MatrixType):
                if (a.type.rows, a.type.cols) != (t.rows, t.cols):
                    raise LamError(f"{self.name}: argument {n}: {a.type} "
                                    f"instead of {t}")
                if a.type.mlir() != t.mlir():
                    a = _cast(a, t)
            vals.append(a)
        names = ", ".join(_size_values(self.sizes, self.name) +
                          [v._use() for v in vals])
        ptypes = ", ".join(["index"] * len(self.sizes) +
                           [t.mlir() for _, t in self.params])
        # the results, then the new values of the parameters it updates
        # (y = bar(M, y) for a bar that overwrites y with the solution)
        types = list(self.results) + list(self.update_types)
        rtypes = [t.mlir() for t in types]
        if not rtypes:
            b.emit(f"func.call @{self.name}({names}) : ({ptypes}) -> ()")
            return None
        v = b.fresh()
        res = v + (f":{len(rtypes)}" if len(rtypes) > 1 else "")
        b.emit(f"{res} = func.call @{self.name}({names}) : ({ptypes}) -> "
               f"({', '.join(rtypes)})")
        if len(rtypes) == 1:
            return Value(v, types[0], b.region.id)
        return tuple(Value(f"{v}#{j}", t, b.region.id)
                     for j, t in enumerate(types))


def function(fn) -> Function:
    """Decorator: a ulm.lam function."""
    return Function(fn)


def module(functions) -> str:
    """The ulm_lam IR of a module with these functions (and everything they
    call)."""
    return module_and_functions(functions)[0]


def module_and_functions(functions):
    """The ulm_lam IR of a module with these functions, and all functions
    in it (the given ones and the ones they call), callees first."""
    prog = _Program()
    outer = _Program.current
    _Program.current = prog
    try:
        todo = list(functions)
        for f in todo:
            if f not in prog.functions:
                prog.functions.append(f)
        # trace: callees register themselves (and externs) while tracing
        i = 0
        while i < len(prog.functions):
            f = prog.functions[i]
            f._text = None
            f.trace()
            i += 1
    finally:
        _Program.current = outer
    # callees before callers
    order, seen = [], set()

    def visit(f):
        if f.name in seen:
            return
        seen.add(f.name)
        for g in prog.functions:
            if g is not f and f"@{g.name}(" in f._text:
                visit(g)
        order.append(f)

    for f in prog.functions:
        visit(f)
    parts = [e.declaration() for e in prog.externs.values()]
    parts += [f._text for f in order]
    return "\n\n".join(parts) + "\n", order
