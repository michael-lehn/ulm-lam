// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize))' | FileCheck %s

// Transposes go to the leaves: (A B)^T = B^T A^T; (A + B)^T = A^T + B^T;
// (s A)^T = s A^T; double transposes cancel.

// CHECK-LABEL: func.func @product
// CHECK-DAG: %[[BT:.*]] = ulm_lam.transpose %arg1
// CHECK-DAG: %[[AT:.*]] = ulm_lam.transpose %arg0
// CHECK: ulm_lam.mul %[[BT]], %[[AT]]
func.func @product(%a: tensor<4x3xf64>, %b: tensor<3x5xf64>) -> tensor<5x4xf64> {
  %p = ulm_lam.mul %a, %b : tensor<4x3xf64>, tensor<3x5xf64> -> tensor<4x5xf64>
  %t = ulm_lam.transpose %p : tensor<4x5xf64> -> tensor<5x4xf64>
  return %t : tensor<5x4xf64>
}

// CHECK-LABEL: func.func @sum
// CHECK-DAG: ulm_lam.transpose %arg0
// CHECK-DAG: ulm_lam.transpose %arg1
// CHECK: ulm_lam.add
func.func @sum(%a: tensor<4x3xf64>, %b: tensor<4x3xf64>) -> tensor<3x4xf64> {
  %s = ulm_lam.add %a, %b : tensor<4x3xf64>, tensor<4x3xf64> -> tensor<4x3xf64>
  %t = ulm_lam.transpose %s : tensor<4x3xf64> -> tensor<3x4xf64>
  return %t : tensor<3x4xf64>
}

// CHECK-LABEL: func.func @twice
// CHECK-NOT: ulm_lam.transpose
// CHECK: return %arg0
func.func @twice(%a: tensor<4x3xf64>) -> tensor<4x3xf64> {
  %t = ulm_lam.transpose %a : tensor<4x3xf64> -> tensor<3x4xf64>
  %u = ulm_lam.transpose %t : tensor<3x4xf64> -> tensor<4x3xf64>
  return %u : tensor<4x3xf64>
}
