"""lib.f.code: the derivation ulm.lam found, as BLAS/LAPACK calls.

    >>> print(lib.gls.code)
    gls(X: n×m full_rank, M: n×n spd, y: n)
      potrf    L1 ← chol(M)                      M = L1 L1ᵀ, Cholesky
      copy     T1 ← X
      trsm     T1 ← L1⁻¹ T1
      ...

It reads the kernels after selection (ulm_lamk on tensors, before
bufferization) and writes each as an assignment in place: B ← L⁻¹ B
overwrites B, as the BLAS call does.  (The Julia frontend's @code_lam,
julia/UlmLam/src/codelam.jl, does the same; Julia's column-major arrays
add the orientation of every value, which Python's row-major ones do not
need.)
"""

import re

from . import frontend as fe

_KERNEL = re.compile(r"^(%[\w#]+)(?::\d+)?(?:, %\w+)* = ulm_lamk\.(\w+)")
_DEFINED = re.compile(r"^(%[\w#]+)(?::\d+)?(?:, %\w+)* = ")


def _operands(line, key):
    m = re.search(key + r"\(([^:)]*)", line)
    return [x.strip() for x in m.group(1).split(",") if x.strip()] if m else []


def _attr(line, a):
    m = re.search(a + r"\((\w+)\)", line)
    return m.group(1) if m else ""


def _accumulate(beta, c, ac, prod):
    """beta C + alpha P as text: C + P, C − P, b C + a P."""
    if beta == "0":
        return ac + prod
    if beta == "-1" and not ac.startswith("-"):
        return ac + prod + " − " + c
    c = ("" if beta == "1" else beta + " ") + c
    return c + " − " + ac[1:] + prod if ac.startswith("-") else \
        c + " + " + ac + prod


class _Renderer:
    def __init__(self):
        self.names = {}
        self.consts = {}
        self.counter = {}
        self.lines = []
        self.params = set()
        self.ifname = {}   # an scf.if (line) -> the name of its result
        self.forced = {}   # a line yielded by an scf.if -> that if's line
        self.k = 0         # the line being rendered
        self.lastuse = {}  # the last line that uses a value

    def fresh(self, c):
        self.counter[c] = self.counter.get(c, 0) + 1
        return f"{c}{self.counter[c]}"

    def name(self, v):
        if v not in self.names:
            self.names[v] = self.fresh("T") if v.startswith("%") else v
        return self.names[v]

    def scal(self, v):
        return self.consts.get(v, self.names.get(v, v))

    def coef(self, v):
        c = self.scal(v)
        return "" if c == "1" else "-" if c == "-1" else c + " "

    def emit(self, indent, kernel, formula, note=""):
        self.lines.append((indent + kernel.ljust(9) + formula.ljust(34) +
                           note).rstrip())

    def result(self, res, out, prefix="T", fresh=False):
        """The result of a kernel writing into out: in place (out's name),
        or a new name (a copy of a parameter, a factor, a new matrix)."""
        if self.k in self.forced and prefix == "T":
            self.names[res] = self.ifname[self.forced[self.k]]
        elif (fresh or out in self.params or prefix != "T"
              or self.lastuse.get(out, 0) > self.k):
            # (a value still used later keeps its name: a new one here)
            self.names[res] = self.fresh(prefix)
        else:
            self.names[res] = self.name(out)
        return self.names[res]


def _tr(s, t):
    return s + "ᵀ" if t else s


def _renumber(text):
    seen, count = {}, {}

    def sub(m):
        k = m.group(0)
        if k not in seen:
            p = m.group(1)
            count[p] = count.get(p, 0) + 1
            seen[k] = f"{p}{count[p]}"
        return seen[k]
    return re.sub(r"\b(LDL|LU|T|L|z)(\d+)(?![0-9A-Za-z_])", sub, text)


def render(fn: fe.Function, selected: str, infos: dict) -> str:
    """The derivation of `fn` from the IR after kernel selection."""
    i = selected.find(f"func.func @{fn.name}(")
    if i < 0:
        raise fe.LamError(f"no function {fn.name} in the IR")
    body = [line.strip() for line in selected[i:].split("\n")]
    r = _Renderer()
    args = re.findall(r"(%arg\d+):", body[0])
    labels = list(fn.sizes) + [n for n, _ in fn.params]
    for a, lab in zip(args, labels):
        r.names[a] = lab
        r.params.add(a)
    empties = set()
    # parameters the function may overwrite (passed on to an inout callee
    # without a copy)
    inout = {m.group(1) for m in re.finditer(r"(%arg\d+): [^{,]*\{([^}]*)\}",
                                             body[0])
             if "ulm_lam.inout" in m.group(2)}
    for k, line in enumerate(body):
        d_ = _DEFINED.match(line)
        for m in re.finditer(r"(%[\w#]+)", line):
            if not (d_ and m.group(1) == d_.group(1)):
                r.lastuse[m.group(1)] = k
    # the values an scf.if yields are its result: named as it
    stack = []
    for k, line in enumerate(body):
        if "scf.if" in line:
            stack.append(("if", k, k))
        elif "scf.for" in line:
            stack.append(("for", k, k))
        elif line.startswith("} else {") and stack:
            kind, at, _ = stack[-1]
            stack[-1] = (kind, at, k)
        elif line == "}" and stack:
            stack.pop()
        elif line.startswith("scf.yield") and stack and stack[-1][0] == "if":
            _, at, start = stack[-1]
            for y in line[len("scf.yield"):].split(" : ")[0].split(","):
                y = y.strip()
                for j in range(k - 1, start - 1, -1):
                    m = _DEFINED.match(body[j])
                    if m and m.group(1) == y:
                        r.forced[j] = at
                        break

    def tdesc(t):
        if isinstance(t, fe.MatrixType):
            shape = f"{t.rows}" if t.cols == 1 else f"{t.rows}×{t.cols}"
            props = sorted(p for p in getattr(t, "props", ()) or ())
            return shape + (" " + ",".join(props) if props else "") + (
                " inout" if getattr(t, "intent", "in") == "inout" else "")
        return "Float" if isinstance(t, fe.FloatType) else "Index"
    r.lines.append(f"{fn.name}(" + ", ".join(
        f"{n}: {tdesc(t)}" for n, t in fn.params) + ")")
    ind, depth = "  ", 1
    for k, line in enumerate(body):
        if k == 0:
            continue
        r.k = k
        if line == "}" and depth == 1:
            break
        d = ind * depth
        m = re.match(r"^(%[\w#]+) = arith.constant ([-\d.e+]+)", line)
        if m:
            v = float(m.group(2))
            r.consts[m.group(1)] = str(int(v)) if v.is_integer() else str(v)
            continue
        m = re.match(r"^(%[\w#]+) = arith\.(\w+) (%[\w#]+), (%[\w#]+)", line)
        if m:
            ops = {"divf": "/", "mulf": "*", "addf": "+", "subf": "-"}
            if m.group(2) in ops:
                x, y = r.scal(m.group(3)), r.scal(m.group(4))
                if m.group(2) == "mulf" and y == "-1":
                    r.consts[m.group(1)] = "-" + x
                elif m.group(2) == "mulf" and x == "-1":
                    r.consts[m.group(1)] = "-" + y
                else:
                    r.consts[m.group(1)] = f"({x} {ops[m.group(2)]} {y})"
            continue
        if "tensor.empty" in line:
            m = _DEFINED.match(line)
            if m:
                empties.add(m.group(1))
            continue
        m = re.match(r"^(?:(%[\w#:]+) = )?scf\.for (%\w+) = \S+ to (%[\w#]+) "
                     r"step \S+(?: iter_args\((.*)\) ->.*)? \{", line)
        if m:
            r.names[m.group(2)] = "i"
            if m.group(4):
                res = m.group(1).split(":")[0]
                for j, pair in enumerate(m.group(4).split(", ")):
                    a, init = [x.strip() for x in pair.split(" = ")]
                    n0 = r.name(init)
                    if init in r.params and n0.endswith("0") and len(n0) > 1:
                        n = n0[:-1]
                    elif init in r.params:
                        n = r.fresh("T")
                    else:
                        n = n0
                    if init in r.params:
                        r.emit(d, "copy", f"{n} ← {n0}",
                               "the loop's state, in place")
                    r.names[a] = n
                    r.names[f"{res}#{j}"] = n
                    if j == 0:
                        r.names[res] = n
            r.lines.append(d + f"for i = 0:{r.name(m.group(3))}-1")
            depth += 1
            continue
        if "scf.if" in line:
            m = _DEFINED.match(line)
            r.ifname[k] = r.fresh("T")
            if m:
                r.names[m.group(1)] = r.ifname[k]
            r.lines.append(d + "if cheaper   (the sizes are known at run "
                           "time: both plans are generated)")
            depth += 1
            continue
        if line.startswith("} else {"):
            r.lines.append(ind * (depth - 1) + "else")
            continue
        if line == "}":
            depth -= 1
            continue
        if line.startswith("scf.yield"):
            continue
        m = re.match(r"^(?:(%[\w#]+) = )?(?:func\.)?call @(\w+)\(([^)]*)\)",
                     line)
        if m:
            # a ulm.lam function (defined in the module) or a callback
            defined = f"func.func @{m.group(2)}(" in selected
            note = "a ulm.lam function" if defined else "your Python function"
            # an inout parameter of the callee gets a copy of a value the
            # caller must not overwrite (a parameter of its own, or a value
            # it still uses): bufferization inserts that copy
            ops_ = [x.strip() for x in m.group(3).split(",") if x.strip()]
            if defined:
                sig = selected[selected.index(f"func.func @{m.group(2)}("):]
                sig = sig[:sig.index("{\n")]
                cargs = re.findall(r"%arg\d+: [^{,]*(?:\{([^}]*)\})?", sig)
                for j, x in enumerate(ops_):
                    if j < len(cargs) and "ulm_lam.inout" in cargs[j] and (
                            (x in r.params and x not in inout)
                            or r.lastuse.get(x, 0) > k):
                        c_ = r.fresh("T")
                        r.emit(d, "copy", f"{c_} ← {r.name(x)}",
                               f"{m.group(2)} may overwrite it")
                        r.names[x + "@copy"] = c_
                        ops_[j] = x + "@copy"
            sizes = set(fn.sizes)
            a = [r.names[x] if x.endswith("@copy") else r.name(x)
                 for x in ops_]
            a = [x for x in a if x not in sizes]
            if m.group(1):
                v = r.fresh("z")
                r.names[m.group(1)] = v
                r.emit(d, "call", f"{v} ← {m.group(2)}({', '.join(a)})",
                       note)
            else:
                r.emit(d, "call", f"{m.group(2)}({', '.join(a)})", note)
            continue
        m = re.match(r"^(%[\w#]+) = bufferization\.materialize_in_destination "
                     r"(%[\w#]+) in (?:writable )?(%[\w#]+)", line)
        if m:
            src, dst = m.group(2), m.group(3)
            r.names[m.group(1)] = r.name(dst)
            if r.name(src) != r.name(dst):
                r.emit(d, "copy", f"{r.name(dst)} ← {r.name(src)}")
            continue
        if line.startswith("return"):
            vs = [x.strip() for x in line[6:].split(" : ")[0].split(",")
                  if x.strip()]
            r.lines.append(ind + "return " + ", ".join(r.name(v) for v in vs))
            continue
        m = _KERNEL.match(line)
        if not m:
            continue
        res, op = m.group(1), m.group(2)
        ins, outs = _operands(line, "ins"), _operands(line, "outs")
        out = outs[0] if outs else ""
        new = out in empties

        def T(a):
            return _attr(line, a) == "trans"
        alpha = r.coef(_operands(line, "alpha")[0]) \
            if _operands(line, "alpha") else ""
        beta = r.scal(_operands(line, "beta")[0]) \
            if _operands(line, "beta") else "0"
        if op in ("potrf", "getrf"):
            a = r.name(out)
            sym = op == "getrf" and "symmetric" in line  # L D Lᵀ
            lname = r.result(res, out, "L" if op == "potrf" else
                             "LDL" if sym else "LU")
            r.names[res + "#0"] = lname
            note = (f"{a} = L D Lᵀ, Bunch–Kaufman" if sym else
                    f"P {a} = L U, LU" if op == "getrf" else
                    f"{a} = {lname} {lname}ᵀ, Cholesky"
                    if _attr(line, "uplo") == "lower" else
                    f"{a} = {lname}ᵀ {lname}, Cholesky")
            fn_ = "chol" if op == "potrf" else "ldlt" if sym else "lu"
            r.emit(d, "sytrf" if sym else op, f"{lname} ← {fn_}({a})", note)
        elif op == "gels":
            a, bb = outs
            r.names[res] = r.names[res + "#0"] = r.name(a)
            r.names[res + "#1"] = r.name(bb)
            r.emit(d, "gels", f"{r.name(bb)} ← lstsq({r.name(a)}, "
                   f"{r.name(bb)})", "QR; the solution in the leading rows")
        elif op in ("gemm", "gemv", "gemmtr"):
            a, b = ins
            ta = T("trans_a") if op != "gemv" else T("trans")
            tb = T("trans_b") if op != "gemv" else False
            old = r.name(out) if beta != "0" else ""
            c = r.result(res, out, fresh=new)
            rhs = _accumulate(beta, old, alpha,
                              f"{_tr(r.name(a), ta)} {_tr(r.name(b), tb)}")
            r.emit(d, op, f"{c} ← {rhs}",
                   "symmetric: one triangle only" if op == "gemmtr" else "")
        elif op == "syrk":
            a = r.name(ins[0])
            t = T("trans")
            old = r.name(out) if beta != "0" else ""
            c = r.result(res, out, fresh=new)
            r.emit(d, "syrk", f"{c} ← " + _accumulate(
                beta, old, alpha, f"{_tr(a, t)} {_tr(a, not t)}"),
                "one triangle only")
        elif op in ("symm", "symv"):
            a, b = ins
            c = r.result(res, out, fresh=new)
            left = _attr(line, "side") != "right"
            g = _tr(r.name(b), T("trans_b"))
            r.emit(d, op, f"{c} ← {alpha}" + (f"{r.name(a)} {g}" if left
                                               else f"{g} {r.name(a)}"),
                   f"{r.name(a)} symmetric")
        elif op in ("trsm", "trsv", "getrs", "trmm"):
            a = r.name(ins[0])
            old = r.name(out)
            b = r.result(res, out)
            t = T("trans")
            f = (a + "ᵀ" if t else a) if op == "trmm" else \
                a + ("⁻ᵀ" if t else "⁻¹")
            al = alpha if op in ("trsm", "trmm") else ""
            left = _attr(line, "side") != "right"
            kname = "sytrs" if op == "getrs" and "symmetric" in line else op
            r.emit(d, kname, f"{b} ← {al}" + (f"{f} {old}" if left
                                               else f"{old} {f}"),
                   f"{a} triangular" if op == "trmm" else "")
        elif op == "gecopy":
            b = r.result(res, out, fresh=True)
            r.emit(d, "copy", f"{b} ← {_tr(r.name(ins[0]), T('trans'))}")
        elif op == "adddiag":
            old = r.name(out)
            a = r.result(res, out)
            r.emit(d, "adddiag", f"{a} ← {old} + {alpha}I")
        elif op == "gescal":
            old = r.name(out)
            a = r.result(res, out)
            r.emit(d, "scal", f"{a} ← {alpha}{old}")
        elif op == "geaxpy":
            old = r.name(out)
            y = r.result(res, out)
            r.emit(d, "axpy", f"{y} ← " + _accumulate(
                "1", old, alpha, r.name(ins[0])))
        elif op == "sycomplete":
            old = r.name(out)
            a = r.result(res, out)
            r.emit(d, "symm.", f"{a} ← sym({old})",
                   "copy one triangle to the other")
        else:
            a = r.result(res, out)
            r.emit(d, op, f"{a} ← {op}(" +
                   ", ".join(r.name(x) for x in ins) + ")")
    checks = sorted(k for k in infos if k > 0)
    if checks:
        r.lines.append("if a factorisation fails, the call raises "
                       "lam.InfoError with")
        for k in checks:
            msg = re.sub(r"\bv(\d+)\b",
                         lambda m: labels[int(m.group(1)) - 1]
                         if int(m.group(1)) <= len(labels) else m.group(0),
                         infos[k])
            r.lines.append(f"  info {k}:  {msg}")
    return _renumber("\n".join(r.lines))
