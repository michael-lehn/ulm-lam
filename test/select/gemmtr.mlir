// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// F P F^T + Q with P symmetric: the last product computes one triangle
// (gemmtr, accumulated onto Q), completed where the full matrix is needed.

// CHECK-LABEL: func.func @fpf
// CHECK: ulm_lamk.gemm
// CHECK: ulm_lamk.gemmtr uplo(lower) trans_a(none) trans_b(trans)
// CHECK: ulm_lamk.sycomplete uplo(lower)
func.func @fpf(%f: tensor<5x5xf64>, %p: tensor<5x5xf64, #ulm_lam.props<symmetric>>, %q: tensor<5x5xf64, #ulm_lam.props<spsd>>) -> tensor<5x5xf64> {
  %ft = ulm_lam.transpose %f : tensor<5x5xf64> -> tensor<5x5xf64>
  %m = ulm_lam.mul %f, %p, %ft : tensor<5x5xf64>, tensor<5x5xf64, #ulm_lam.props<symmetric>>, tensor<5x5xf64> -> tensor<5x5xf64>
  %s = ulm_lam.add %m, %q : tensor<5x5xf64>, tensor<5x5xf64, #ulm_lam.props<spsd>> -> tensor<5x5xf64>
  return %s : tensor<5x5xf64>
}
