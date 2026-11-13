// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' | FileCheck %s

// Plans found by comparing the selection with an e-graph (optimal DAG
// extraction, experiments/egraph/lamp_fuzz.py).

// LoRA's adapted weight applied to a vector, (W + B A) x = W x + B (A x):
// the sum is distributed (for all sampled run-time sizes), B A is never
// formed.
// CHECK-LABEL: func.func @lora
// CHECK-NOT: ulm_lamk.gemm
// CHECK-COUNT-3: ulm_lamk.gemv
// CHECK-NOT: ulm_lamk.gemm
// CHECK: return

// A solve inside a sum, (C + B^-T) D u: distributed, B^-T is applied to a
// vector (getrs), never formed.
// CHECK-LABEL: func.func @solve_in_sum
// CHECK: ulm_lamk.getrf
// CHECK-NOT: ulm_lamk.getri
// CHECK: ulm_lamk.getrs
// CHECK: return

// A Gram matrix inside a longer product, A A^T B: syrk (one triangle), read
// by symm.
// CHECK-LABEL: func.func @gram_left
// CHECK: ulm_lamk.syrk
// CHECK: ulm_lamk.symm
// CHECK: return

// An inverse used as a value: potrf + potri (2 n^3 / 3 flops) instead of
// solving with the identity.
// CHECK-LABEL: func.func @inv_spd
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.potri
// CHECK-NOT: ulm_lamk.trsm
// CHECK: return

func.func @lora(%s.n: index, %s.r: index, %p.W: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.B: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]}, %p.A: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [1, 0]}, %p.x: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [0]}) {
  %v1 = ulm_lam.mul %p.B, %p.A : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v2 = ulm_lam.add %p.W, %v1 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v3 = ulm_lam.mul %v2, %p.x : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  return %v3 : tensor<?x1xf64>
}

func.func @solve_in_sum(%p.A: tensor<1000x1000xf64, #ulm_lam.props<spd>> {ulm_lam.in}, %p.B: tensor<1000x1000xf64, #ulm_lam.props<non_singular>> {ulm_lam.in}, %p.C: tensor<1000x1000xf64> {ulm_lam.in}, %p.D: tensor<1000x200xf64> {ulm_lam.in}, %p.u: tensor<200x1xf64> {ulm_lam.in}) -> (tensor<1000x1xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.transpose %p.B : tensor<1000x1000xf64, #ulm_lam.props<non_singular>> -> tensor<1000x1000xf64>
  %v2 = ulm_lam.inverse %v1 : tensor<1000x1000xf64> -> tensor<1000x1000xf64>
  %v3 = ulm_lam.add %p.C, %v2 : tensor<1000x1000xf64>, tensor<1000x1000xf64> -> tensor<1000x1000xf64>
  %v4 = ulm_lam.mul %p.A, %v3 : tensor<1000x1000xf64, #ulm_lam.props<spd>>, tensor<1000x1000xf64> -> tensor<1000x1000xf64>
  %v5 = ulm_lam.mul %v4, %p.D : tensor<1000x1000xf64>, tensor<1000x200xf64> -> tensor<1000x200xf64>
  %v6 = ulm_lam.mul %v5, %p.u : tensor<1000x200xf64>, tensor<200x1xf64> -> tensor<1000x1xf64>
  return %v6 : tensor<1000x1xf64>
}

func.func @gram_left(%p.A: tensor<30x200xf64> {ulm_lam.in}, %p.B: tensor<30x30xf64> {ulm_lam.in}) -> (tensor<30x30xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.transpose %p.A : tensor<30x200xf64> -> tensor<200x30xf64>
  %v2 = ulm_lam.mul %p.A, %v1 : tensor<30x200xf64>, tensor<200x30xf64> -> tensor<30x30xf64>
  %v3 = ulm_lam.mul %v2, %p.B : tensor<30x30xf64>, tensor<30x30xf64> -> tensor<30x30xf64>
  return %v3 : tensor<30x30xf64>
}

func.func @inv_spd(%p.A: tensor<200x200xf64, #ulm_lam.props<spd>> {ulm_lam.in}) -> (tensor<200x200xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.inverse %p.A : tensor<200x200xf64, #ulm_lam.props<spd>> -> tensor<200x200xf64>
  return %v1 : tensor<200x200xf64>
}


// Costing a distribution must not factorise: S^-1 (Y - H Xb) is costed
// before S (a sum asserted spd) is selected; a Cholesky factorisation
// created then would keep S's type of before the selection (the bug:
// "potrf result must have the type of its init").
// CHECK-LABEL: func.func @ensemble_kalman
// CHECK: ulm_lamk.potrf
// CHECK: return

func.func @ensemble_kalman(%p.B: tensor<1000x1000xf64, #ulm_lam.props<spd>> {ulm_lam.in}, %p.H: tensor<200x1000xf64> {ulm_lam.in}, %p.R: tensor<200x200xf64, #ulm_lam.props<spd>> {ulm_lam.in}, %p.Y: tensor<200x50xf64> {ulm_lam.in}, %p.Xb: tensor<1000x50xf64> {ulm_lam.in}) -> (tensor<1000x50xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.mul %p.H, %p.B : tensor<200x1000xf64>, tensor<1000x1000xf64, #ulm_lam.props<spd>> -> tensor<200x1000xf64>
  %v2 = ulm_lam.transpose %p.H : tensor<200x1000xf64> -> tensor<1000x200xf64>
  %v3 = ulm_lam.mul %v1, %v2 : tensor<200x1000xf64>, tensor<1000x200xf64> -> tensor<200x200xf64>
  %v4 = ulm_lam.add %v3, %p.R : tensor<200x200xf64>, tensor<200x200xf64, #ulm_lam.props<spd>> -> tensor<200x200xf64>
  %v5 = ulm_lam.assume %v4 : tensor<200x200xf64> -> tensor<200x200xf64, #ulm_lam.props<spd>>
  %v6 = ulm_lam.mul %p.B, %v2 : tensor<1000x1000xf64, #ulm_lam.props<spd>>, tensor<1000x200xf64> -> tensor<1000x200xf64>
  %v7 = ulm_lam.inverse %v5 : tensor<200x200xf64, #ulm_lam.props<spd>> -> tensor<200x200xf64>
  %v8 = ulm_lam.mul %v6, %v7 : tensor<1000x200xf64>, tensor<200x200xf64> -> tensor<1000x200xf64>
  %v9 = ulm_lam.mul %p.H, %p.Xb : tensor<200x1000xf64>, tensor<1000x50xf64> -> tensor<200x50xf64>
  %v10 = arith.constant -1.00000000000000000e+00 : f64
  %v11 = ulm_lam.scale %v10, %v9 : f64, tensor<200x50xf64> -> tensor<200x50xf64>
  %v12 = ulm_lam.add %p.Y, %v11 : tensor<200x50xf64>, tensor<200x50xf64> -> tensor<200x50xf64>
  %v13 = ulm_lam.mul %v8, %v12 : tensor<1000x200xf64>, tensor<200x50xf64> -> tensor<1000x50xf64>
  %v14 = ulm_lam.add %p.Xb, %v13 : tensor<1000x50xf64>, tensor<1000x50xf64> -> tensor<1000x50xf64>
  return %v14 : tensor<1000x50xf64>
}

// A sum of given matrices is never distributed (its parentheses may be set
// for accuracy): A (C - D) x forms C - D.  (Distributing would not be
// cheaper here either; the rule makes it a guarantee.)
// CHECK-LABEL: func.func @sacred
// CHECK: ulm_lamk.geaxpy
// CHECK-COUNT-2: ulm_lamk.gemv
// CHECK-NOT: ulm_lamk.gemv
// CHECK: return

func.func @sacred(%p.A: tensor<1000x1000xf64> {ulm_lam.in}, %p.B: tensor<1000x1000xf64> {ulm_lam.in}, %p.C: tensor<1000x30xf64> {ulm_lam.in}, %p.D: tensor<1000x30xf64> {ulm_lam.in}, %p.x: tensor<30x1xf64> {ulm_lam.in}) -> (tensor<1000x1xf64> {ulm_lam.out}) {
  %v1 = arith.constant -1.00000000000000000e+00 : f64
  %v2 = ulm_lam.scale %v1, %p.D : f64, tensor<1000x30xf64> -> tensor<1000x30xf64>
  %v3 = ulm_lam.add %p.C, %v2 : tensor<1000x30xf64>, tensor<1000x30xf64> -> tensor<1000x30xf64>
  %v4 = ulm_lam.mul %p.A, %v3 : tensor<1000x1000xf64>, tensor<1000x30xf64> -> tensor<1000x30xf64>
  %v5 = ulm_lam.mul %v4, %p.x : tensor<1000x30xf64>, tensor<30x1xf64> -> tensor<1000x1xf64>
  return %v5 : tensor<1000x1xf64>
}
