// Formula (1) in a loop: a measurement update in every time step.
//
//   for i = 0, ..., n-1:
//       y_i = observe(i)                                  2000 sensors
//       b_i = (H^T R^-1 H)^-1 H^T R^-1 y_i                200 states
//       record(i, b_i)
//
// H is the observation matrix (full rank), R the covariance of the sensor
// noise (spd, correlated).  This is the measurement update of a Kalman
// filter in information form,
//       b = (P^-1 + H^T R^-1 H)^-1 (P^-1 b_prior + H^T R^-1 y),
// with an uninformative prior (P^-1 = 0), i.e. generalised least squares
// in every step.  Only y_i changes from step to step.
//
// The formula is written inside the loop, as on paper.  loop-invariant-
// code-motion hoists the invariant ulm_lam ops (they are Pure), kernel
// selection factorises R and S = H^T R^-1 H right after their definitions,
// i.e. before the loop, and the loop body is left with O(n^2) solves.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!H  = tensor<2000x200xf64, #fr>
!Ht = tensor<200x2000xf64, #fr>
!R  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!S  = tensor<200x200xf64, #spd>
!b  = tensor<200x1xf64>

func.func private @observe(index) -> !y
func.func private @record(index, !b)

func.func @estimate(%H: !H, %R: !R, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %y  = func.call @observe(%i) : (index) -> !y
    %Ht = ulm_lam.transpose %H : !H -> !Ht
    %Ri = ulm_lam.inverse %R : !R -> !R
    %S  = ulm_lam.mul %Ht, %Ri, %H : !Ht, !R, !H -> !S
    %Si = ulm_lam.inverse %S : !S -> !S
    %b  = ulm_lam.mul %Si, %Ht, %Ri, %y : !S, !Ht, !R, !y -> !b
    func.call @record(%i, %b) : (index, !b) -> ()
  }
  return
}
