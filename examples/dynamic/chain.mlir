// A chain whose best parenthesisation depends on the sizes:
//
//   D = A B C,   A: n x k,  B: k x m,  C: m x p
//
// (A B) C costs 2nkm + 2nmp, A (B C) costs 2kmp + 2nkp.  The sizes are
// known only at run time, so the generated code computes both costs and
// runs the cheaper plan.

!A = tensor<?x?xf64>

func.func @chain(%n: index, %k: index, %m: index, %p: index,
                 %A: !A {ulm_lam.dims = [0, 1]},
                 %B: !A {ulm_lam.dims = [1, 2]},
                 %C: !A {ulm_lam.dims = [2, 3]}) -> (!A {ulm_lam.dims = [0, 3]}) {
  %D = ulm_lam.mul %A, %B, %C : !A, !A, !A -> !A
  return %D : !A
}
