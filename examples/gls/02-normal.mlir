// Stage 2: after ulm-lam-normalize, cse, ulm-lam-infer-properties.
//
// Products are flattened (no parenthesisation is fixed any more), the
// duplicated X^T and M^-1 are shared, and the flattened product
// X^T M^-1 X is now recognised as spd:
//   A^T P A  with P spd and A full (column) rank  =>  spd.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!X  = tensor<2000x200xf64, #fr>
!Xt = tensor<200x2000xf64, #fr>
!M  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!S  = tensor<200x200xf64, #spd>
!b  = tensor<200x1xf64>

func.func @gls(%X: !X, %M: !M, %y: !y) -> !b {
  %Xt = ulm_lam.transpose %X : !X -> !Xt
  %Mi = ulm_lam.inverse %M : !M -> !M
  %S  = ulm_lam.mul %Xt, %Mi, %X : !Xt, !M, !X -> !S
  %Si = ulm_lam.inverse %S : !S -> !S
  %b  = ulm_lam.mul %Si, %Xt, %Mi, %y : !S, !Xt, !M, !y -> !b
  return %b : !b
}
