// Kernel selection cases, each checked numerically by driver.cc against a
// naive evaluation.  n = 64, k = 32.
//
//   lu        A^-1 B             A general        getrf + getrs
//   lu_t      A^-T B                              getrs (trans)
//   upper     U^-1 B             U upper_tri      trsm (upper, left)
//   right     B L^-1             L lower_tri      trsm (right)
//   right_t   B L^-T                              trsm (right, trans)
//   chol_vec  S^-1 x             S spd, x vector  potrf + 2 trsv
//   sum       Y + 2 Z - W                         gescal/geaxpy
//   mixed     A^-1 B C + D                        getrs + gemm (beta = 1)

!sq  = tensor<64x64xf64>
!lo  = tensor<64x64xf64, #ulm_lam.props<lower_tri>>
!up  = tensor<64x64xf64, #ulm_lam.props<upper_tri>>
!spd = tensor<64x64xf64, #ulm_lam.props<spd>>
!rk  = tensor<64x32xf64>
!kr  = tensor<32x64xf64>
!v   = tensor<64x1xf64>

func.func @lu(%A: !sq, %B: !rk) -> !rk {
  %Ai = ulm_lam.inverse %A : !sq -> !sq
  %X = ulm_lam.mul %Ai, %B : !sq, !rk -> !rk
  return %X : !rk
}

func.func @lu_t(%A: !sq, %B: !rk) -> !rk {
  %At = ulm_lam.transpose %A : !sq -> !sq
  %Ati = ulm_lam.inverse %At : !sq -> !sq
  %X = ulm_lam.mul %Ati, %B : !sq, !rk -> !rk
  return %X : !rk
}

func.func @upper(%U: !up, %B: !rk) -> !rk {
  %Ui = ulm_lam.inverse %U : !up -> !up
  %X = ulm_lam.mul %Ui, %B : !up, !rk -> !rk
  return %X : !rk
}

func.func @right(%B: !kr, %L: !lo) -> !kr {
  %Li = ulm_lam.inverse %L : !lo -> !lo
  %X = ulm_lam.mul %B, %Li : !kr, !lo -> !kr
  return %X : !kr
}

func.func @right_t(%B: !kr, %L: !lo) -> !kr {
  %Lt = ulm_lam.transpose %L : !lo -> !up
  %Lti = ulm_lam.inverse %Lt : !up -> !up
  %X = ulm_lam.mul %B, %Lti : !kr, !up -> !kr
  return %X : !kr
}

func.func @chol_vec(%S: !spd, %x: !v) -> !v {
  %Si = ulm_lam.inverse %S : !spd -> !spd
  %r = ulm_lam.mul %Si, %x : !spd, !v -> !v
  return %r : !v
}

func.func @sum(%Y: !sq, %Z: !sq, %W: !sq) -> !sq {
  %two = arith.constant 2.0 : f64
  %m1 = arith.constant -1.0 : f64
  %Z2 = ulm_lam.scale %two, %Z : f64, !sq -> !sq
  %Wm = ulm_lam.scale %m1, %W : f64, !sq -> !sq
  %r = ulm_lam.add %Y, %Z2, %Wm : !sq, !sq, !sq -> !sq
  return %r : !sq
}

func.func @mixed(%A: !sq, %B: !sq, %C: !rk, %D: !rk) -> !rk {
  %Ai = ulm_lam.inverse %A : !sq -> !sq
  %P = ulm_lam.mul %Ai, %B, %C : !sq, !sq, !rk -> !rk
  %r = ulm_lam.add %D, %P : !rk, !rk -> !rk
  return %r : !rk
}
