// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' | FileCheck %s

// The Kalman update x + K (z - H x), P - K H P with K = P H^T S^-1 (from
// experiments/egraph/lamp_ir.py).  K H P is expanded to (H P)^T S^-1 (H P)
// (syrk); then K is needed only for K r, and expanding that as well,
// W^T (L^-1 r) with W = L^-1 (H P), saves the trsm that would form K.

// CHECK-LABEL: func.func @kalman_update
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.trsm
// CHECK-NOT: ulm_lamk.trsm
// CHECK: ulm_lamk.trsv
// CHECK-NOT: ulm_lamk.trsm
// CHECK: ulm_lamk.syrk
// CHECK-NOT: ulm_lamk.trsm
// CHECK: return

func.func @kalman_update(%p.H: tensor<100x1000xf64> {ulm_lam.in}, %p.P: tensor<1000x1000xf64, #ulm_lam.props<spd>> {ulm_lam.in}, %p.R: tensor<100x100xf64, #ulm_lam.props<spd>> {ulm_lam.in}, %p.z: tensor<100x1xf64> {ulm_lam.in}, %p.x: tensor<1000x1xf64> {ulm_lam.in}) -> (tensor<1000x1xf64> {ulm_lam.out}, tensor<1000x1000xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.mul %p.H, %p.P : tensor<100x1000xf64>, tensor<1000x1000xf64, #ulm_lam.props<spd>> -> tensor<100x1000xf64>
  %v2 = ulm_lam.transpose %p.H : tensor<100x1000xf64> -> tensor<1000x100xf64>
  %v3 = ulm_lam.mul %v1, %v2 : tensor<100x1000xf64>, tensor<1000x100xf64> -> tensor<100x100xf64>
  %v4 = ulm_lam.add %v3, %p.R : tensor<100x100xf64>, tensor<100x100xf64, #ulm_lam.props<spd>> -> tensor<100x100xf64>
  %v5 = ulm_lam.assume %v4 : tensor<100x100xf64> -> tensor<100x100xf64, #ulm_lam.props<spd>>
  %v6 = ulm_lam.mul %p.P, %v2 : tensor<1000x1000xf64, #ulm_lam.props<spd>>, tensor<1000x100xf64> -> tensor<1000x100xf64>
  %v7 = ulm_lam.inverse %v5 : tensor<100x100xf64, #ulm_lam.props<spd>> -> tensor<100x100xf64>
  %v8 = ulm_lam.mul %v6, %v7 : tensor<1000x100xf64>, tensor<100x100xf64> -> tensor<1000x100xf64>
  %v9 = ulm_lam.mul %p.H, %p.x : tensor<100x1000xf64>, tensor<1000x1xf64> -> tensor<100x1xf64>
  %v10 = arith.constant -1.00000000000000000e+00 : f64
  %v11 = ulm_lam.scale %v10, %v9 : f64, tensor<100x1xf64> -> tensor<100x1xf64>
  %v12 = ulm_lam.add %p.z, %v11 : tensor<100x1xf64>, tensor<100x1xf64> -> tensor<100x1xf64>
  %v13 = ulm_lam.mul %v8, %v12 : tensor<1000x100xf64>, tensor<100x1xf64> -> tensor<1000x1xf64>
  %v14 = ulm_lam.add %p.x, %v13 : tensor<1000x1xf64>, tensor<1000x1xf64> -> tensor<1000x1xf64>
  %v15 = ulm_lam.mul %v8, %p.H : tensor<1000x100xf64>, tensor<100x1000xf64> -> tensor<1000x1000xf64>
  %v16 = ulm_lam.mul %v15, %p.P : tensor<1000x1000xf64>, tensor<1000x1000xf64, #ulm_lam.props<spd>> -> tensor<1000x1000xf64>
  %v17 = ulm_lam.scale %v10, %v16 : f64, tensor<1000x1000xf64> -> tensor<1000x1000xf64>
  %v18 = ulm_lam.add %p.P, %v17 : tensor<1000x1000xf64, #ulm_lam.props<spd>>, tensor<1000x1000xf64> -> tensor<1000x1000xf64>
  return %v14, %v18 : tensor<1000x1xf64>, tensor<1000x1000xf64>
}
