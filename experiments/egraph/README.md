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

## Back into the compiler

The Kalman finding is now a rule of ulm.lam's kernel selection
(`expand-factors`, on by default): a factor of a product that is itself a
computed product (K in K H P) may be replaced by the factors of that
product, if the longer chain is cheaper by the generalised matrix chain.
In K H P = P Hᵀ S⁻¹ H P the chain then reads (H P)ᵀ L⁻ᵀ L⁻¹ (H P), which
the chain algorithm already knew as a Gram matrix (as Xᵀ M⁻¹ X in GLS); H P,
the factorisation of S and L⁻¹ H P are reused from computing K.  Two small
changes made it work: a symmetric matrix counts as its own transpose when
the chain looks for Zᵀ … Z, and the stored triangle of a syrk result is
completed in tiles (`sycomplete`; element-wise, its transposed writes cost
more than syrk saves for n = 3000).

The Kalman filter (predict and update, Julia's OpenBLAS with 4 threads on an
Apple M5, the two variants compiled side by side and timed alternately,
best of 7):

| n, m | without | with | faster | asymmetry of P after 20 steps (without / with) |
|:--|--:|--:|--:|:--|
| 1000, 100 | 29.14 ms/step | 28.73 ms/step | 1.4 % | 3.4e-16 / 0 |
| 1000, 300 | 38.56 | 36.76 | 4.7 % | 1.2e-15 / 0 |
| 2000, 200 | 219.5 | 214.8 | 2.1 % | 4.7e-16 / 0 |
| 3000, 300 | 759.8 | 759.4 | 0.1 % | 5.2e-16 / 0 |

The gain is real but small: the update is a few percent of a step, in which
F P Fᵀ dominates (the 23 % of the experiment are of the measurement update
alone), and at n = 3000 the completion of the triangle eats what syrk saves.
The covariance, however, is now exactly symmetric in every step; without the
rule it drifts away from symmetry by rounding, a known problem of Kalman
filters.

Next: the experiment driven by ulm.lam's IR (the traced `ulm_lam` functions
instead of hand-written terms) and its plans lowered back to `ulm_lamk`, as
`strategy=egraph`; MLIR's own equality saturation work (eqsat, DialEgg) as
an alternative to egglog.
