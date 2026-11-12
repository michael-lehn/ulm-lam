// Stage 4: after one-shot-bufferize{bufferize-function-boundaries} and
// buffer-results-to-out-params{hoist-static-allocs}.
//
// The arguments are marked bufferization.writable = false by ulm-lam-compile
// (callers' inputs are not clobbered unless they opt in), so the analysis
// inserts exactly the copies Linnea would: M (overwritten by potrf),
// X (overwritten by trsm) and y (overwritten by trsv).  S needs a fresh
// buffer.  c, d and b are written straight into the result buffer, because
// every trsv updates its outs operand in place.

!X = memref<2000x200xf64>
!M = memref<2000x2000xf64>
!y = memref<2000x1xf64>
!S = memref<200x200xf64>
!b = memref<200x1xf64>

func.func @gls(%X: !X, %M: !M, %y: !y, %b: !b) {
  %one  = arith.constant 1.0 : f64
  %zero = arith.constant 0.0 : f64

  %L = memref.alloc() {alignment = 64 : i64} : !M
  memref.copy %M, %L : !M to !M
  %infoL = ulm_lamk.potrf uplo(lower) outs(%L : !M)

  %W = memref.alloc() {alignment = 64 : i64} : !X
  memref.copy %X, %W : !X to !X
  ulm_lamk.trsm side(left) uplo(lower) trans(none) diag(non_unit)
    alpha(%one) ins(%L : !M) outs(%W : !X)

  %z = memref.alloc() {alignment = 64 : i64} : !y
  memref.copy %y, %z : !y to !y
  ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%L : !M) outs(%z : !y)

  %S = memref.alloc() {alignment = 64 : i64} : !S
  ulm_lamk.syrk uplo(lower) trans(trans) alpha(%one) beta(%zero)
    ins(%W : !X) outs(%S : !S)
  %infoS = ulm_lamk.potrf uplo(lower) outs(%S : !S)

  ulm_lamk.gemv trans(trans) alpha(%one) beta(%zero)
    ins(%W, %z : !X, !y) outs(%b : !b)
  ulm_lamk.trsv uplo(lower) trans(none) diag(non_unit) ins(%S : !S) outs(%b : !b)
  ulm_lamk.trsv uplo(lower) trans(trans) diag(non_unit) ins(%S : !S) outs(%b : !b)

  memref.dealloc %S : !S
  memref.dealloc %z : !y
  memref.dealloc %W : !X
  memref.dealloc %L : !M
  return
}
