"""Optimal DAG extraction from an e-graph by an ILP (integer linear program).

egglog's DAG extraction is greedy: it may miss the cheapest plan when two
subterms share a factorisation only in some of their variants (it decided
the ensemble Kalman filter and the triangular sandwich worse than ulm.lam's
selection).  The cheapest DAG is the solution of an ILP (as in the extraction
of egg and of "Tensat", Yang et al., 2021):

  x_n in {0, 1}           e-node n is computed
  sum_{n in root} x_n >= 1
  x_n <= sum_{m in d} x_m for every child class d of n
  t_c >= t_d + 1 - B (1 - x_n)   for n in class c, child class d  (acyclic)
  minimise sum_n flops(n) x_n

A computed e-node is paid once, however often it is used: the DAG cost.
The flops of an e-node are those of lamp_egraph.flops, from the sizes of its
child classes, which the e-graph holds as R(class) and C(class).  The ILP is
solved with HiGHS (scipy.optimize.milp).
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

MAT = "lamp_egraph.Mat"
DEBUG = False


def node_flops(op, kids, size, props=None):
    """ulm.lam's flop count of the kernel of an e-node (None: never chosen),
    from the sizes (rows, cols) of its child classes and their properties
    (props: class -> set of relations)."""
    if op in ("Mat", "T", "Pair"):
        return 0.0
    s = [size[k] for k in kids]
    if op == "Inv":
        # an explicit inverse, when solves are not enough: potrf + potri
        # (n^3), trtri (n^3 / 3), getrf + getri (2 n^3)
        if props is None:
            return None
        n, p = s[0][0], props.get(kids[0], set())
        if "lower" in p:
            return n ** 3 / 3.0
        return float(n) ** 3 if "spd" in p else 2.0 * n ** 3
    if op == "· * ·":
        (m, k), (_, n) = s
        return 2.0 * m * k * n
    if op in ("· + ·", "· - ·"):
        m, n = s[0]
        return 1.0 * m * n
    if op == "AddDiag":
        return float(s[0][0])
    if op == "Gemmtr":
        m, k = s[0]
        return 1.0 * m * m * k
    if op == "Chol":
        return s[0][0] ** 3 / 3.0
    if op == "Lu":
        return 2.0 * s[0][0] ** 3 / 3.0
    if op in ("TrsmL", "TrsmLt"):
        n, k = s[1]
        return 1.0 * n * n * k
    if op == "LuSolve":
        n, k = s[1]
        return 2.0 * n * n * k
    if op == "Syrk":
        k, n = s[0]
        return 1.0 * n * n * k
    if op == "Trmm":
        m, n = s[0]
        return 1.0 * m * n * n
    raise ValueError(f"no flop count for {op}")


def extract(egraph, root="$root", time_limit=30.0, inverses=True):
    """The cheapest DAG of the let `root`: (flops, plan as text, kernels)."""
    data = json.loads(egraph._serialize(split_primitive_outputs=False)
                      .to_json())
    nodes, classes = data["nodes"], data["class_data"]
    cls = {k: v["eclass"] for k, v in nodes.items()}
    rootc = next(c for c, v in classes.items() if v.get("let") == root)

    # the sizes of the matrix classes: R(c) and C(c) are i64 classes
    size: dict[str, list] = {}
    for v in nodes.values():
        if v["op"] in ("R", "C"):
            c = cls[v["children"][0]]
            val = int(v["eclass"].split("-", 1)[1])
            size.setdefault(c, [None, None])[v["op"] == "C"] = val
    size = {c: tuple(rc) for c, rc in size.items()}
    props: dict[str, set] = {}
    for v in nodes.values():
        if v["op"] in ("spd", "lower", "sym", "spsd", "fullrank"):
            props.setdefault(cls[v["children"][0]], set()).add(v["op"])

    # the candidates: e-nodes of matrix classes with a finite cost
    cand, cost, kids = [], [], []
    for k, v in nodes.items():
        if classes[v["eclass"]]["type"] != MAT or v.get("subsumed"):
            continue
        ch = [cls[x] for x in v["children"]
              if classes[cls[x]]["type"] == MAT]
        try:
            f = node_flops(v["op"], ch, size, props if inverses else None)
        except (KeyError, TypeError):
            continue  # a class without sizes: not part of a valid plan
        if f is None:
            continue
        cand.append(k)
        cost.append(f)
        kids.append(ch)
    mats = sorted({v["eclass"] for v in nodes.values()
                   if classes[v["eclass"]]["type"] == MAT})
    ci = {c: i for i, c in enumerate(mats)}
    nx, nt = len(cand), len(mats)
    members: dict[str, list[int]] = {}
    for i, k in enumerate(cand):
        members.setdefault(cls[k], []).append(i)

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def row(entries, lb, ub):
        nonlocal r
        for j, a in entries:
            rows.append(r)
            cols.append(j)
            vals.append(a)
        lo.append(lb)
        hi.append(ub)
        r += 1

    row([(j, 1.0) for j in members.get(rootc, [])], 1.0, np.inf)
    big = nt + 1.0
    for i, k in enumerate(cand):
        c = cls[k]
        for d in set(kids[i]):
            # x_i <= sum of the class d
            row([(i, 1.0)] + [(j, -1.0) for j in members.get(d, [])],
                -np.inf, 0.0)
            # t_c - t_d - B x_i >= 1 - B
            if d == c:
                row([(i, 1.0)], -np.inf, 0.0)  # a node in its own class
            else:
                row([(nx + ci[c], 1.0), (nx + ci[d], -1.0), (i, -big)],
                    1.0 - big, np.inf)
    A = coo_matrix((vals, (rows, cols)), shape=(r, nx + nt))
    scale = max(max(cost), 1.0)
    obj = np.concatenate([np.array(cost) / scale + 1e-9, np.zeros(nt)])
    res = milp(obj, constraints=LinearConstraint(A, lo, hi),
               integrality=np.concatenate([np.ones(nx), np.zeros(nt)]),
               bounds=Bounds(np.zeros(nx + nt),
                             np.concatenate([np.ones(nx),
                                             np.full(nt, big)])),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    if res.x is None or res.status != 0:  # (0: optimal)
        raise RuntimeError(f"ILP: {res.message}")
    chosen = {i for i in range(nx) if res.x[i] > 0.5}
    pick = {}
    for i in chosen:  # one e-node per class (the cheapest chosen one)
        c = cls[cand[i]]
        if c not in pick or cost[i] < cost[pick[c]]:
            pick[c] = i

    # the plan: kernels in topological order, shared ones named once
    names, lines, total = {}, [], 0.0
    visiting = set()
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20 * len(pick) + 1000))

    def show(c):
        if c in names:
            return names[c]
        if c in visiting:  # (the levels' tolerance admitted a cycle)
            if DEBUG:
                for cc in visiting:
                    j = pick[cc]
                    print("  ", cc, nodes[cand[j]]["op"], kids[j],
                          res.x[j], res.x[nx + ci[cc]])
            raise RuntimeError("ILP: cyclic solution")
        visiting.add(c)
        i = pick[c]
        op = nodes[cand[i]]["op"]
        args = [show(d) for d in kids[i]]
        if op == "Mat":
            ch = nodes[cand[i]]["children"]
            names[c] = nodes[ch[0]]["op"].strip('"')
            return names[c]
        if op == "T":
            names[c] = f"{args[0]}ᵀ" if len(args[0]) <= 2 else \
                f"({args[0]})ᵀ"
            return names[c]
        if op == "Pair":
            names[c] = f"({args[0]}, {args[1]})"
            return names[c]
        nonlocal total
        total += cost[i]
        names[c] = f"%{len(lines)}"
        sym = {"· * ·": "mul", "· + ·": "add", "· - ·": "sub"}.get(op, op)
        lines.append((sym, f"{names[c]} = {sym}({', '.join(args)})",
                      cost[i]))
        return names[c]

    result = show(rootc)
    return total, result, lines
