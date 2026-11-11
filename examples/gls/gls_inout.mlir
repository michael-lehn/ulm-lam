// GLS with parameter intents (the normal form of stage 2):
//
//   gls_inout(X: in, M: inout, y: inout) -> b: out
//
// As in LAPACK, the caller allows the function to overwrite M and y: their
// contents on exit are unspecified ("destroyed").  The Cholesky
// factorisation of M and the solve with y then work in place on the
// caller's matrices; the copies of M and y, and their 32 MB of workspace,
// disappear.  X is {ulm_lam.in} (the default): it is copied for the in-place
// solve W = L^-1 X.  b is {ulm_lam.out} (the default for results): the
// caller's buffer, written by gemv and the two solves.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!X  = tensor<2000x200xf64, #fr>
!Xt = tensor<200x2000xf64, #fr>
!M  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!S  = tensor<200x200xf64, #spd>
!b  = tensor<200x1xf64>

func.func @gls_inout(%X: !X {ulm_lam.in}, %M: !M {ulm_lam.inout}, %y: !y {ulm_lam.inout})
    -> (!b {ulm_lam.out}) {
  %Xt = ulm_lam.transpose %X : !X -> !Xt
  %Mi = ulm_lam.inverse %M : !M -> !M
  %S  = ulm_lam.mul %Xt, %Mi, %X : !Xt, !M, !X -> !S
  %Si = ulm_lam.inverse %S : !S -> !S
  %b  = ulm_lam.mul %Si, %Xt, %Mi, %y : !S, !Xt, !M, !y -> !b
  return %b : !b
}
