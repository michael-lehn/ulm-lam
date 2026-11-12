# Equality saturation for the Linear Algebra Mapping Problem

A first experiment with [egglog](https://github.com/egraphs-good/egglog-python)
(`make egraph`): the algebra of `ulm_lam` and the kernels of `ulm_lamk` as
constructors of one e-graph, general rules between them, properties as
relations derived by rules, and the cheapest term extracted with ulm.lam's
flop counts for the sizes of the problem.

The rules are general, not written for a problem: associativity;
transposes ((AB)ᵀ = BᵀAᵀ, Sᵀ = S for spd S); solves for products with
inverses (Cholesky for spd matrices, LU otherwise, from the left and from
the right); Xᵀ L⁻ᵀ = (L⁻¹X)ᵀ; Gram matrices (ZᵀZ and WWᵀ are syrk);
S = L Lᵀ for spd S; A L with L triangular is trmm; A S Aᵀ with S symmetric
is symmetric (gemmtr); properties: Cholesky factors are lower triangular,
L⁻¹X keeps the full rank of X, ZᵀZ is spd for Z of full rank, A S Aᵀ is
spsd for spd S, and spsd + spd is spd.

Flops (`lamp_egraph.py`):

| problem | e-graph, tree extraction | e-graph, DAG extraction | ulm.lam (`strategy=gmc`) |
|:--|--:|--:|--:|
| GLS, n = 2000, m = 50 | 8.21e9 | 2.88e9 | 2.88e9 |
| GP, n = 4000, m = 1000 | 8.40e10 | 4.14e10 | 4.14e10 |
| F P Fᵀ, P spd, n = 3000 | 6.30e10 | 6.30e10 | 6.30e10 |
| Kalman step, n = 1000, m = 100 | 9.52e8 | **3.31e8** | 4.30e8 |
| Q Kᵀ V, N = 4096, d = 64 | 6.71e7 | 6.71e7 | 6.71e7 |

What it shows:

- With DAG extraction (a shared subterm, such as a factorisation used by
  two solves, is paid once) the e-graph finds ulm.lam's plans from general
  rules, including Linnea's variant of F P Fᵀ through the Cholesky factor
  of P, which ulm.lam needs a rule of its own for.
- For the Kalman step it finds a plan with 23 % fewer flops than ulm.lam:
  with S = L Lᵀ, K H P = P Hᵀ S⁻¹ H P = Wᵀ W with W = L⁻¹ (H P), which K
  needs anyway, so P − K H P is a syrk (n²m flops) instead of a gemm
  (2n²m), and symmetric by construction, which also matters numerically.
  ulm.lam's rules compute K first and see K (H P) only as a product.
- Tree extraction pays a shared subterm at every use and therefore misses
  plans that share a factorisation (GLS, GP, the Kalman step).  DAG
  extraction is the right model, but egglog's is greedy; optimal DAG
  extraction is NP-hard (ILP extractors exist).
- The e-graphs stay small here (at most 3212 nodes and table rows,
  saturation in 10 ms), but associativity on long chains and sums lets
  them grow exponentially.
- Extraction needs the sizes.  ulm.lam decides between plans at run time
  when the sizes are known only then; with an e-graph that means symbolic
  costs (polynomials in the sizes) and a set of plans that are optimal in
  different regions, instead of one extraction.

Next: the syrk downdate of the Kalman step as a rule of `strategy=gmc`;
the experiment driven by ulm.lam's IR (the traced `ulm_lam` functions
instead of hand-written terms) and its plans lowered back to `ulm_lamk`, as
`strategy=egraph`; MLIR's own equality saturation work (eqsat, DialEgg) as
an alternative to egglog.
