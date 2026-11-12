// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize))' | FileCheck %s

// Identity factors of a product are dropped; products are flattened.

// CHECK-LABEL: func.func @drop
// CHECK-NOT: ulm_lam.identity
// CHECK: %[[P:.*]] = ulm_lam.mul %arg0, %arg1 :
// CHECK: return %[[P]]
func.func @drop(%a: tensor<4x4xf64>, %b: tensor<4x2xf64>) -> tensor<4x2xf64> {
  %i = ulm_lam.identity : tensor<4x4xf64, #ulm_lam.props<identity>>
  %p = ulm_lam.mul %a, %i : tensor<4x4xf64>, tensor<4x4xf64, #ulm_lam.props<identity>> -> tensor<4x4xf64>
  %q = ulm_lam.mul %p, %b : tensor<4x4xf64>, tensor<4x2xf64> -> tensor<4x2xf64>
  return %q : tensor<4x2xf64>
}

// CHECK-LABEL: func.func @flatten
// CHECK: ulm_lam.mul %arg0, %arg1, %arg2 :
func.func @flatten(%a: tensor<4x3xf64>, %b: tensor<3x5xf64>, %c: tensor<5x2xf64>) -> tensor<4x2xf64> {
  %p = ulm_lam.mul %a, %b : tensor<4x3xf64>, tensor<3x5xf64> -> tensor<4x5xf64>
  %q = ulm_lam.mul %p, %c : tensor<4x5xf64>, tensor<5x2xf64> -> tensor<4x2xf64>
  return %q : tensor<4x2xf64>
}
