// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// A triangular factor: the other factor copied, then trmm in place.

// CHECK-LABEL: func.func @left
// CHECK: ulm_lamk.gecopy
// CHECK: ulm_lamk.trmm side(left) uplo(lower) trans(none) diag(non_unit)
// CHECK-NOT: ulm_lamk.gemm
func.func @left(%l: tensor<4x4xf64, #ulm_lam.props<lower_tri>>, %b: tensor<4x3xf64>) -> tensor<4x3xf64> {
  %p = ulm_lam.mul %l, %b : tensor<4x4xf64, #ulm_lam.props<lower_tri>>, tensor<4x3xf64> -> tensor<4x3xf64>
  return %p : tensor<4x3xf64>
}

// CHECK-LABEL: func.func @right_transposed_unit
// CHECK: ulm_lamk.trmm side(right) uplo(upper) trans(trans) diag(unit)
func.func @right_transposed_unit(%c: tensor<3x4xf64>, %u: tensor<4x4xf64, #ulm_lam.props<upper_tri, unit_diag>>) -> tensor<3x4xf64> {
  %ut = ulm_lam.transpose %u : tensor<4x4xf64, #ulm_lam.props<upper_tri, unit_diag>> -> tensor<4x4xf64, #ulm_lam.props<lower_tri, unit_diag>>
  %p = ulm_lam.mul %c, %ut : tensor<3x4xf64>, tensor<4x4xf64, #ulm_lam.props<lower_tri, unit_diag>> -> tensor<3x4xf64>
  return %p : tensor<3x4xf64>
}
