// Stage 3: after ulm-lam-select-kernels and ulm-lam-strip-properties.
//
// The derivation found by the search (FLOPs for n = 2000, m = 200):
//
//   M = L L^T            potrf   n^3/3     2.67e9
//   W = L^-1 X           trsm    n^2 m     8.00e8
//   z = L^-1 y           trsv    n^2       4.00e6
//   S = W^T W            syrk    m^2 n     8.00e7   (spd: W full rank)
//   S = K K^T            potrf   m^3/3     2.67e6
//   c = W^T z            gemv    2 n m     8.00e5
//   b = K^-T (K^-1 c)    2x trsv 2 m^2     8.00e4
//                                total     3.55e9
//
// Kernel results have BLAS semantics: potrf/syrk define the lower
// triangle only; the strict upper triangle keeps the init's values
// (undefined for tensor.empty).  Every consumer reads the lower triangle.

!X = tensor<2000x200xf64>
!M = tensor<2000x2000xf64>
!y = tensor<2000x1xf64>
!S = tensor<200x200xf64>
!b = tensor<200x1xf64>

func.func @gls(%X: !X, %M: !M, %y: !y) -> !b {
  %one  = arith.constant 1.0 : f64
  %zero = arith.constant 0.0 : f64

  %L = ulm_lamk.potrf uplo(lower) checked outs(%M : !M) -> !M
  %W = ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit)
         alpha(%one) ins(%L : !M) outs(%X : !X) -> !X
  %z = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit)
         ins(%L : !M) outs(%y : !y) -> !y

  %e0 = tensor.empty() : !S
  %S  = ulm_lamk.syrk uplo(lower) trans(trans) alpha(%one) beta(%zero)
          ins(%W : !X) outs(%e0 : !S) -> !S
  %K  = ulm_lamk.potrf uplo(lower) checked outs(%S : !S) -> !S

  %e1 = tensor.empty() : !b
  %c  = ulm_lamk.gemv trans(trans) alpha(%one) beta(%zero)
          ins(%W, %z : !X, !y) outs(%e1 : !b) -> !b
  %d  = ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit)
          ins(%K : !S) outs(%c : !b) -> !b
  %b  = ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit)
          ins(%K : !S) outs(%d : !b) -> !b
  return %b : !b
}
