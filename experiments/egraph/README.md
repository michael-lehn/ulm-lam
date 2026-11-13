# Equality saturation for the Linear Algebra Mapping Problem

A first experiment with [egglog](https://github.com/egraphs-good/egglog-python)
(`make egraph`): the algebra of `ulm_lam` and the kernels of `ulm_lamk` as
constructors of one e-graph, general rules between them, properties as
relations derived by rules, and the cheapest term extracted with ulm.lam's
flop (floating-point operation) counts for the sizes of the problem.

The rules are general, not written for a problem: associativity;
transposes ((AB)ᵀ = BᵀAᵀ, Sᵀ = S for spd (symmetric positive definite) S); solves for products with
inverses (Cholesky for spd matrices, LU (lower-upper factorisation) otherwise, from the left and from
the right); Xᵀ L⁻ᵀ = (L⁻¹X)ᵀ; Gram matrices (ZᵀZ and WWᵀ are syrk);
S = L Lᵀ for spd S; A L with L triangular is trmm; A S Aᵀ with S symmetric
is symmetric (gemmtr); properties: Cholesky factors are lower triangular,
L⁻¹X keeps the full rank of X, ZᵀZ is spd for Z of full rank, A S Aᵀ is
spsd (symmetric positive semidefinite) for spd S, and spsd + spd is spd.

Flops (`lamp_egraph.py`):

| problem | e-graph, tree extraction | e-graph, DAG (directed acyclic graph) extraction | ulm.lam (`strategy=gmc`, the generalised matrix chain) |
|:--|--:|--:|--:|
| GLS (generalised least squares), n = 2000, m = 50 | 8.21e9 | 2.88e9 | 2.88e9 |
| GP (Gaussian process), n = 4000, m = 1000 | 8.40e10 | 4.14e10 | 4.14e10 |
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
  extraction is NP-hard (NP: nondeterministic polynomial time; ILP extractors, ILP for integer linear program, exist).
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

## Driven by ulm.lam's IR (intermediate representation)

`lamp_ir.py` (`make egraph-ir`) does the same for problems written with the
Python frontend, with static sizes, instead of hand-written terms: the
traced `ulm_lam` IR becomes the e-graph term (properties of the parameters
and of `lam.assume` become facts), and the same IR goes through ulm.lam's
kernel selection (`ulm-lam-opt`), whose `ulm_lamk` kernels are counted with
the same flop formulas.  No count is written by hand any more, so any
function can be checked; where the e-graph is cheaper, the selection misses
something.  The problems: the showcases and Linnea's application problems
(Barthels, Psarras, Bientinesi, 2021) that the frontend can express.

| problem | ulm.lam | e-graph, greedy DAG | e-graph, optimal DAG (ILP) | ILP / ulm.lam |
|:--|--:|--:|--:|--:|
| gls | 2.88e+09 | 2.88e+09 | 2.88e+09 | 1.00 |
| gp | 4.14e+10 | 4.14e+10 | 4.14e+10 | 1.00 |
| fpft | 1.87e+10 | 1.87e+10 | 1.87e+10 | 1.00 |
| kalman_update | 3.21e+08 | 3.22e+08 | 3.22e+08 | 1.00 |
| attention | 6.71e+07 | 6.71e+07 | 6.71e+07 | 1.00 |
| ridge | 1.24e+08 | 1.24e+08 | 1.24e+08 | 1.00 |
| normal_equations | 1.24e+08 | 1.24e+08 | 1.24e+08 | 1.00 |
| optimization | 4.44e+08 | 4.46e+08 | 4.44e+08 | 1.00 |
| ensemble_kalman | 4.87e+08 | 5.25e+08 | 4.87e+08 | 1.00 |
| image_restoration | 2.34e+09 | 2.34e+09 | 2.34e+09 | 1.00 |
| two_solves | 3.39e+09 | 3.39e+09 | 3.39e+09 | 1.00 |
| triangular_sandwich | 1.6e+10 | 1.6e+10 | 1.6e+10 | 1.00 |
| chain_vector | 1.3e+07 | 1.3e+07 | 1.3e+07 | 1.00 |
| gram_of_solve | 3.09e+09 | 3.09e+09 | 3.09e+09 | 1.00 |
| kalman_gain_cov | 2.3e+09 | 2.31e+09 | 2.31e+09 | 1.00 |

Its first finding, now fixed: for the Kalman update `x + K (z − H x)`,
`P − K H P` the e-graph was 3 % cheaper.  `expand-factors` expanded
`K H P` (the syrk of the first finding), but decided for `K r` on its own,
where K exists already and one `gemv` is cheapest; that K is then needed
for nothing else, and forming it costs a `trsm`, was not seen.  Now, if all
other uses of a computed factor are expanded, its forming cost counts for
the expansion, and the expanded chain must not reuse the factor itself:
`K r` becomes `Wᵀ (L⁻¹ r)` with `W = L⁻¹ (H P)` from the syrk, one `trsv`
and one `gemv`, and K is never formed (`test/select/expand_unformed.mlir`).
With sizes known only at run time this exposed a second weakness: the
selection compared the plans of its size samples by their whole tables,
including intervals no plan uses, and so decided at run time between two
identical branches, in which `syrk` is not used.  Plans are now compared by
the intervals they reach.

The e-graph is an oracle only if its extraction is optimal and its model
knows what the selection knows.  Two changes made it one:

- Optimal DAG extraction (`ilp_extract.py`): egglog's DAG extraction is
  greedy and decided the ensemble Kalman filter and the optimization
  problem worse than ulm.lam.  The cheapest DAG is the solution of an ILP
  (one binary variable per e-node, one acyclicity
  level per e-class, as in the extraction of Tensat), solved with HiGHS
  through SciPy in at most a few seconds for these e-graphs.
- The model: `A + αI` adds to the diagonal (`AddDiag`, n flops), spd
  matrices are spsd, `(A⁻¹)ᵀ = (Aᵀ)⁻¹` in both directions (the right solve
  of the triangular sandwich), distributivity (`(A + B) C = A C + B C`),
  and explicit inverses where solves are not enough (potri, getri, trtri
  flops).

With both, the e-graph agrees with ulm.lam on all 15 problems: the table
above.

## Random problems

`lamp_fuzz.py` (`make egraph-fuzz`) generates problems at random: products
of 1 to 4 factors with transposes and inverses (of spd, non-singular and
lower triangular matrices), operands used more than once, sums inside
products, one or two results, sizes from 1 (vectors) to 1000.  Each problem
is Python source for the frontend, reproducible from its seed.  The first
200 problems had 33 where the e-graph was cheaper, by up to a factor of
166.  Three causes accounted for most of them, all fixed in the selection:

- **Distributivity.**  A sum as a factor of a product was always formed:
  `(E + A B C D) F G u` computed the 1000 × 200 sum with three `gemm`s
  before applying it to a vector.  Now a product with a sum as a factor is
  distributed when that is cheaper by the generalised matrix chain, for all
  sampled run-time sizes (`distribute` in the selection, option
  `distribute`).  That is the adapted weight of LoRA (low-rank adaptation), `(W + B A) x = W x + B (A
  x)`, and a solve inside a sum, `(C + B⁻ᵀ) x = C x + B⁻ᵀ x`, which then needs
  no explicit inverse.
- **Gram matrices inside longer products.**  `A Aᵀ B` was costed with
  `syrk` for `A Aᵀ` but emitted as `gemm`.  Now the Gram matrix is a `syrk`
  (one triangle), read by `symm`.
- **Explicit inverses.**  An inverse used as a value (a result, a term of a
  sum: the information form `(P⁻¹ + Hᵀ R⁻¹ H)⁻¹`) was a solve with the
  identity, 7n³/3 flops for spd matrices.  Now it is `potrf` + `potri`
  (2n³/3) or `getrf` + `getri` (2n³), new kernels of `ulm_lamk` for ulmBLAS
  and for LAPACK.  The information form goes from 5.4e6 to 1.7e6 flops at
  n = 100.

After these, 16 and 20 findings remain in two runs of 200 problems
(seeds 1 and 1001), most within 10 %.  The larger ones:

- `B + A Aᵀ` with B not symmetric: `syrk` into a temporary and an addition
  (n²k + n²) instead of one `gemm` with β = 1 (2n²k).
- The inverse of a triangular matrix (`trtri`, n³/3) instead of a solve
  with the identity (n³); `trtri` leaves the other triangle, which must
  then be zeroed.
- Distribution where the terms share sub-products (`A B (E + C D B E) Eᵀ`):
  the cost comparison of `distribute` does not see the reuse of `B E`.
- A product of inverses only (`A⁻¹ B⁻¹`) still has no plan in ulm.lam (an
  error), and `(A⁻¹)ᵀ` of an LU factorisation is still a solve with the
  identity.

## The Woodbury identity

A rule beyond the selection's: `(I + ZᵀZ)⁻¹ C = C − Zᵀ (I + ZZᵀ)⁻¹ Z C`
(scales are not modelled), an N × N instead of a p × p factorisation for Z
of size N × p.  It applies to its own result again, so it is a ruleset of
its own, run once between the rounds of the other rules; otherwise the
e-graph grows to millions of e-classes, the ILP's acyclicity constraints
need a big-M of that size, and HiGHS's integrality tolerance then admits
cycles (the extraction now rejects a non-optimal or cyclic solution).

The selection got the same rewrite (`woodbury`): `inv(αI + β ZᵀZ)` used
only in products is generated both ways, in an `scf.if` on N < p, and the
Woodbury form is distributed over the products, so that no p × p matrix is
formed (`test/select/woodbury.mlir`, `python/examples/laplace.py`: the
last-layer Laplace approximation).  The e-graph agrees with the choice in
both regimes:

| problem | ulm.lam | e-graph (ILP) |
|:--|--:|--:|
| laplace_wide, N = 500, p = 4000 | 2.09e+09 | 2.07e+09 |
| laplace_tall, N = 4000, p = 500 | 1.12e+09 | 1.12e+09 |

The 1 % in the wide case: the e-graph computes the mean from products it
forms for the covariance and the factorisation anyway (`Φ* Φᵀ`, `Φ Φᵀ`),
the selection with matrix-vector products of its own.

Next: plans lowered back to `ulm_lamk` (`strategy=egraph`), which needs
symbolic costs when sizes are known only at run time; MLIR's
(Multi-Level Intermediate Representation) own equality saturation work
(eqsat, DialEgg) as an alternative to egglog.
