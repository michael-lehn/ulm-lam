"""Random problems: ulm.lam's kernel selection against the e-graph.

lamp_ir.py compares the selection with the e-graph (optimal DAG extraction)
on hand-picked problems.  This generates problems at random: products of
matrices, transposes and inverses (of spd, non-singular and lower triangular
matrices), with operands used more than once (which is where factorisations
are shared), sums and differences, and one or two results.  Each problem is
written as Python source for the frontend, so a finding can be reproduced
from its seed and pasted as a test.

    build/venv-egglog/bin/python experiments/egraph/lamp_fuzz.py [count] [seed]

A problem where the e-graph is cheaper than ulm.lam is a finding: the
selection misses a plan that the e-graph's general rules find.
"""

from __future__ import annotations

import random
import sys

import lamp_ir
from lamp_ir import egraph_plan, ir_to_term, ulmlam_plan
from ulm import lam
from ulm.lam import Matrix, Vector  # noqa: F401  (used by exec)

SIZES = (1, 30, 200, 1000)


class Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.params = {}  # name -> (rows, cols, prop)

    def param(self, r, c, prop=None):
        """An operand of r x c (with prop): an existing one (or the
        transpose of one) with probability 1/2, else a new one."""
        rng = self.rng
        same = [(n, False) for n, (a, b, p) in self.params.items()
                if (a, b) == (r, c) and (prop is None or p == prop)]
        same += [(n, True) for n, (a, b, p) in self.params.items()
                 if (a, b) == (c, r) and (prop is None or p == prop)
                 and r != c and c != 1]
        if same and rng.random() < 0.5:
            n, t = rng.choice(same)
            return f"{n}.T" if t else n
        name = "ABCDEFGHJKMNPQRSTUWXYZ"[len(self.params)] if c != 1 else \
            "uvwxyz"[sum(1 for v in self.params.values() if v[1] == 1)]
        if prop is None and r == c and r > 1:
            prop = rng.choice([None, None, "symmetric", "spd"])
        elif prop is None and r > c:
            prop = rng.choice([None, "full_rank"])
        self.params[name] = (r, c, prop)
        return name

    def inverse(self, n):
        prop = self.rng.choice(["spd", "spd", "non_singular", "lower_tri"])
        p = self.param(n, n, prop)
        return f"{p}.inv" if self.rng.random() < 0.8 else f"{p}.T.inv"

    def chain(self, r, c, depth=0):
        """A product of 1 to 4 factors of r x c in total."""
        rng = self.rng
        k = rng.randint(1, 4)
        dims = [r] + [rng.choice(SIZES[1:]) for _ in range(k - 1)] + [c]
        fs = []
        for i in range(k):
            a, b = dims[i], dims[i + 1]
            if a == b and a > 1 and rng.random() < 0.4:
                fs.append(self.inverse(a))
            elif depth < 1 and b > 1 and a > 1 and rng.random() < 0.15:
                fs.append(f"({self.sum(a, b, depth + 1)})")
            else:
                fs.append(self.param(a, b))
        return " @ ".join(fs)

    def sum(self, r, c, depth=0):
        """A chain, or a matrix plus or minus a chain."""
        rng = self.rng
        t = self.chain(r, c, depth)
        if rng.random() < 0.35:
            t = f"{self.param(r, c)} {rng.choice('+-')} {t}"
        return t

    def problem(self, name):
        rng = self.rng
        r = rng.choice(SIZES[1:])
        c = rng.choice(SIZES)
        outs = [self.sum(r, c)]
        if rng.random() < 0.3:
            outs.append(self.sum(rng.choice(SIZES[1:]), rng.choice(SIZES)))
        args = []
        for n, (a, b, p) in self.params.items():
            if b == 1:
                args.append(f"{n}: Vector({a})")
            else:
                args.append(f"{n}: Matrix({a}, {b}" +
                            (f', "{p}")' if p else ")"))
        return (f"def {name}({', '.join(args)}):\n"
                f"    return {', '.join(outs)}\n")


def run(count, seed):
    findings, gaps, errors = [], [], []
    for i in range(count):
        rng = random.Random(seed + i)
        name = f"p{seed + i}"
        src = Gen(rng).problem(name)
        env = {"Matrix": Matrix, "Vector": Vector}
        try:
            exec(src, env)
            fn = lam.function(env[name])
            text = lam.module([fn])
        except Exception as ex:  # an invalid random problem
            errors.append((name, src, "frontend: " + str(ex)[:150]))
            continue
        try:
            kernels, ulm = ulmlam_plan(text, name)
        except Exception as ex:
            errors.append((name, src, "ulm.lam: " + str(ex)[:300]))
            continue
        try:
            lamp_ir.negated = set()
            term, facts = ir_to_term(text, name)
            _, eg, result, lines, _, _ = egraph_plan(term, facts)
        except Exception as ex:
            errors.append((name, src, "e-graph: " + type(ex).__name__ +
                           " " + str(ex)[:150]))
            continue
        row = (name, src, ulm, eg, kernels, lines, result)
        if eg < 0.999 * ulm:
            findings.append(row)
        elif eg > 1.001 * ulm:
            gaps.append(row)
        print(f"{name}: ulm.lam {ulm:.3g}  e-graph {eg:.3g}  "
              f"{'FINDING' if eg < 0.999 * ulm else ''}", flush=True)
    print(f"\n{count} problems: {len(findings)} findings (e-graph cheaper), "
          f"{len(gaps)} where ulm.lam is cheaper, {len(errors)} errors")
    for name, src, ulm, eg, kernels, lines, result in findings:
        print(f"\n== {name}: ulm.lam {ulm:.3g}, e-graph {eg:.3g} "
              f"({eg / ulm:.2f})\n{src}   ulm.lam: "
              + " ".join(k for k, _ in kernels))
        for _, t, f in lines:
            print(f"      {t:60s} {f:.3g}")
        print(f"      result {result}")
    for name, src, ulm, eg, *_ in gaps:
        print(f"\n-- {name}: ulm.lam {ulm:.3g}, e-graph {eg:.3g} "
              f"({eg / ulm:.2f})\n{src}")
    for name, src, msg in errors:
        print(f"\n!! {name}: {msg}\n{src}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(run(int(a[0]) if a else 50, int(a[1]) if len(a) > 1 else 1))
