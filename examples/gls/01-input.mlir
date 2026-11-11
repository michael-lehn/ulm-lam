// Stage 1: frontend output.
//
//   b = (X^T M^-1 X)^-1 X^T M^-1 y,   M spd, X full rank
//
// The frontend builds binary operations left to right and does not share
// subexpressions.  Result properties are inferred locally by the op
// builders (InferTypeOpInterface): M^-1 is spd, X^T is full rank -- but the
// binary product (X^T M^-1) X is not recognised as spd yet.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!X  = tensor<2000x200xf64, #fr>
!Xt = tensor<200x2000xf64, #fr>
!M  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!T  = tensor<200x2000xf64>
!G  = tensor<200x200xf64>
!b  = tensor<200x1xf64>

func.func @gls(%X: !X, %M: !M, %y: !y) -> !b {
  %0 = ulm_lam.transpose %X : !X -> !Xt
  %1 = ulm_lam.inverse %M : !M -> !M
  %2 = ulm_lam.mul %0, %1 : !Xt, !M -> !T
  %3 = ulm_lam.mul %2, %X : !T, !X -> !G
  %4 = ulm_lam.inverse %3 : !G -> !G
  %5 = ulm_lam.transpose %X : !X -> !Xt
  %6 = ulm_lam.mul %4, %5 : !G, !Xt -> !T
  %7 = ulm_lam.inverse %M : !M -> !M
  %8 = ulm_lam.mul %6, %7 : !T, !M -> !T
  %9 = ulm_lam.mul %8, %y : !T, !y -> !b
  return %9 : !b
}
