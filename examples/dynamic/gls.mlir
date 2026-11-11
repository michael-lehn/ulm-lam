// Formula (1) with sizes known only at run time:
//
//   b = (X^T M^-1 X)^-1 X^T M^-1 y,   X: n x m, M: n x n spd, y: n x 1
//
// The sizes are index parameters; `ulm_lam.dims` ties every dynamic
// dimension of a matrix to one of them (argument positions, one per `?`).

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!X  = tensor<?x?xf64, #fr>
!Xt = tensor<?x?xf64, #fr>
!M  = tensor<?x?xf64, #spd>
!y  = tensor<?x1xf64>
!S  = tensor<?x?xf64, #spd>
!b  = tensor<?x1xf64>

func.func @gls(%n: index, %m: index,
               %X: !X {ulm_lam.dims = [0, 1]},
               %M: !M {ulm_lam.dims = [0, 0]},
               %y: !y {ulm_lam.dims = [0]}) -> (!b {ulm_lam.dims = [1]}) {
  %Xt = ulm_lam.transpose %X : !X -> !Xt
  %Mi = ulm_lam.inverse %M : !M -> !M
  %S  = ulm_lam.mul %Xt, %Mi, %X : !Xt, !M, !X -> !S
  %Si = ulm_lam.inverse %S : !S -> !S
  %b  = ulm_lam.mul %Si, %Xt, %Mi, %y : !S, !Xt, !M, !y -> !b
  return %b : !b
}
