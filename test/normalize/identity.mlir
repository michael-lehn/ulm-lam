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

// An argument declared as the identity is dropped as well: the rule looks at
// the properties, not at the operation that produced the value.
// CHECK-LABEL: func.func @drop_argument
// CHECK-NOT: ulm_lam.mul
// CHECK: return %arg1
func.func @drop_argument(%i: tensor<3x3xf64, #ulm_lam.props<identity>>, %a: tensor<3x4xf64>) -> tensor<3x4xf64> {
  %p = ulm_lam.mul %i, %a : tensor<3x3xf64, #ulm_lam.props<identity>>, tensor<3x4xf64> -> tensor<3x4xf64>
  return %p : tensor<3x4xf64>
}

// A shared subproduct is not flattened, so it is computed once.
// CHECK-LABEL: func.func @shared
// CHECK: %[[P:.*]] = ulm_lam.mul %arg1, %arg2 :
// CHECK: %[[Q:.*]] = ulm_lam.mul %arg0, %[[P]] :
// CHECK: return %[[Q]], %[[P]]
func.func @shared(%a: tensor<3x4xf64>, %b1: tensor<4x5xf64>, %b2: tensor<5x2xf64>) -> (tensor<3x2xf64>, tensor<4x2xf64>) {
  %p = ulm_lam.mul %b1, %b2 : tensor<4x5xf64>, tensor<5x2xf64> -> tensor<4x2xf64>
  %q = ulm_lam.mul %a, %p : tensor<3x4xf64>, tensor<4x2xf64> -> tensor<3x2xf64>
  return %q, %p : tensor<3x2xf64>, tensor<4x2xf64>
}
