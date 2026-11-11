// Functions with a workspace calling functions with a workspace.
//
//   gls(X, M, y) -> b           formula (1), a library function
//   estimate_all(X, M, n):      for i: y_i = observe(i)
//                                      b_i = gls(X, M, y_i)
//                                      postprocess(i, b_i)
//
// gls needs temporaries, so it takes a workspace.  estimate_all needs
// temporaries (y_i, b_i) and calls gls and postprocess, which take
// workspaces too: postprocess is an external function declared with
// {ulm_lam.workspace} (its last two parameters are work and workspace_info).
// A query of estimate_all therefore returns its own requirement plus the
// answers of gls and postprocess to their queries, as a LAPACK routine
// that calls other routines does; in the compute phase every callee gets
// its part of estimate_all's workspace.
//
// gls is called as a function: it factorises M in every call.  The
// inlined formula of examples/kalman/estimate.mlir factorises M once.

#fr  = #ulm_lam.props<full_rank>
#spd = #ulm_lam.props<spd>

!X  = tensor<2000x200xf64, #fr>
!Xt = tensor<200x2000xf64, #fr>
!M  = tensor<2000x2000xf64, #spd>
!y  = tensor<2000x1xf64>
!S  = tensor<200x200xf64, #spd>
!b  = tensor<200x1xf64>

func.func private @observe(index) -> !y
func.func private @postprocess(index, !b) attributes {ulm_lam.workspace}

func.func @gls(%X: !X, %M: !M, %y: !y) -> !b {
  %Xt = ulm_lam.transpose %X : !X -> !Xt
  %Mi = ulm_lam.inverse %M : !M -> !M
  %S  = ulm_lam.mul %Xt, %Mi, %X : !Xt, !M, !X -> !S
  %Si = ulm_lam.inverse %S : !S -> !S
  %b  = ulm_lam.mul %Si, %Xt, %Mi, %y : !S, !Xt, !M, !y -> !b
  return %b : !b
}

func.func @estimate_all(%X: !X, %M: !M, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %y = func.call @observe(%i) : (index) -> !y
    %b = func.call @gls(%X, %M, %y) : (!X, !M, !y) -> !b
    func.call @postprocess(%i, %b) : (index, !b) -> ()
  }
  return
}
