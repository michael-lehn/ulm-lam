// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// Values used outside the algebra: a scaled result (gescal at the end), an
// inverse as a result (a factorisation and a solve with the identity, formed
// by gescal 0 and adddiag 1), the identity as a result.

// CHECK-LABEL: func.func @scaled_solve
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.trsv
// CHECK: ulm_lamk.gescal
func.func @scaled_solve(%s: tensor<4x4xf64, #ulm_lam.props<spd>>, %x: tensor<4x1xf64>) -> tensor<4x1xf64> {
  %two = arith.constant 2.0 : f64
  %si = ulm_lam.inverse %s : tensor<4x4xf64, #ulm_lam.props<spd>> -> tensor<4x4xf64>
  %p = ulm_lam.mul %si, %x : tensor<4x4xf64>, tensor<4x1xf64> -> tensor<4x1xf64>
  %r = ulm_lam.scale %two, %p : f64, tensor<4x1xf64> -> tensor<4x1xf64>
  return %r : tensor<4x1xf64>
}

// CHECK-LABEL: func.func @inverse
// CHECK: ulm_lamk.getrf
// CHECK: ulm_lamk.gescal
// CHECK: ulm_lamk.adddiag
// CHECK: ulm_lamk.getrs
func.func @inverse(%a: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %ai = ulm_lam.inverse %a : tensor<4x4xf64> -> tensor<4x4xf64>
  return %ai : tensor<4x4xf64>
}

// CHECK-LABEL: func.func @identity
// CHECK: ulm_lamk.gescal
// CHECK: ulm_lamk.adddiag
func.func @identity(%n: index) -> tensor<?x?xf64> {
  %i = ulm_lam.identity(%n, %n) : tensor<?x?xf64, #ulm_lam.props<identity>>
  %j = ulm_lam.assume %i : tensor<?x?xf64, #ulm_lam.props<identity>> -> tensor<?x?xf64>
  return %j : tensor<?x?xf64>
}
