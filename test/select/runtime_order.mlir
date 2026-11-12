// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels))' | FileCheck %s

// Q K^T V with sizes known only at run time: both orders, chosen by an
// scf.if on the flop counts.

// CHECK-LABEL: func.func @attention
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK-COUNT-2: ulm_lamk.gemm
// CHECK: } else {
// CHECK-COUNT-2: ulm_lamk.gemm
func.func @attention(%n: index, %d: index, %e: index, %q: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]}, %k: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]}, %v: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 2]}) -> (tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [0, 2]}) {
  %kt = ulm_lam.transpose %k : tensor<?x?xf64> -> tensor<?x?xf64>
  %p = ulm_lam.mul %q, %kt, %v : tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %p : tensor<?x?xf64>
}
