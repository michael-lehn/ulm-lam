// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// Generalised least squares (X^T M^-1 X)^-1 X^T M^-1 y: one Cholesky
// factorisation of M for both inverses, X^T M^-1 X as W^T W with
// W = L^-1 X (trsm, syrk), a second Cholesky factorisation, no inverse.

// CHECK-LABEL: func.func @gls
// CHECK-NOT: ulm_lam.inverse
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.trsm
// CHECK: ulm_lamk.syrk
// CHECK: ulm_lamk.potrf
// CHECK-NOT: ulm_lamk.gemm
// CHECK: return
func.func @gls(%n: index, %m: index, %x: tensor<?x?xf64, #ulm_lam.props<full_rank>> {ulm_lam.in, ulm_lam.dims = [0, 1]}, %mm: tensor<?x?xf64, #ulm_lam.props<spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [1]}) {
  %xt = ulm_lam.transpose %x : tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64>
  %mi = ulm_lam.inverse %mm : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %g = ulm_lam.mul %xt, %mi, %x : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64, #ulm_lam.props<full_rank>> -> tensor<?x?xf64>
  %gi = ulm_lam.inverse %g : tensor<?x?xf64> -> tensor<?x?xf64>
  %b = ulm_lam.mul %gi, %xt, %mi, %y : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  return %b : tensor<?x1xf64>
}
