// Formula (1) in a loop with a covariance that changes in every step:
//
//   for i = 0, ..., n-1:
//       R_i = update_covariance(i, R_{i-1})               2000 x 2000, spd
//       y_i = observe(i)
//       b_i = (H^T R_i^-1 H)^-1 H^T R_i^-1 y_i
//       record(i, b_i)
//
// The noise covariance is re-estimated in every step (an adaptive filter,
// or sensors whose noise drifts).  R is carried from step to step and
// updated in place by the external update_covariance(i, R): its result is
// marked {ulm_lam.inout = 1}, so R is passed by reference and stays in one
// buffer (the caller owns it).
//
// Now R_i^-1 changes in every step: the Cholesky factorisation of R_i,
// W = L_i^-1 H, S = W^T W and its factorisation must run in every step.
// What can stay outside is only what depends on H alone.  Kernel selection
// still shares the factorisation of R_i between both occurrences of
// R_i^-1 and reuses W for H^T R_i^-1 y_i.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!H  = tensor<2000x200xf64, #fr>
!Ht = tensor<200x2000xf64, #fr>
!R  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!S  = tensor<200x200xf64, #spd>
!b  = tensor<200x1xf64>

// update_covariance(i, R) updates R in place (the result is argument 1)
func.func private @update_covariance(index, !R) -> (!R {ulm_lam.inout = 1 : i64})
func.func private @observe(index) -> !y
func.func private @record(index, !b)

func.func @estimate_adaptive(%H: !H, %R0: !R, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %Rn = scf.for %i = %c0 to %n step %c1 iter_args(%R = %R0) -> !R {
    %Ri = func.call @update_covariance(%i, %R) : (index, !R) -> !R
    %y  = func.call @observe(%i) : (index) -> !y
    %Ht = ulm_lam.transpose %H : !H -> !Ht
    %Ri_inv = ulm_lam.inverse %Ri : !R -> !R
    %S  = ulm_lam.mul %Ht, %Ri_inv, %H : !Ht, !R, !H -> !S
    %Si = ulm_lam.inverse %S : !S -> !S
    %b  = ulm_lam.mul %Si, %Ht, %Ri_inv, %y : !S, !Ht, !R, !y -> !b
    func.call @record(%i, %b) : (index, !b) -> ()
    scf.yield %Ri : !R
  }
  return
}
