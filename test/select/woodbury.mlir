// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' | FileCheck %s

// Bayesian linear regression (the Laplace approximation of a last layer),
// A = alpha I + beta Phi^T Phi with Phi N x p, used in products only: both
// algorithms are generated, chosen at run time by N < p.  The Woodbury
// branch factorises the N x N matrix alpha I + beta Phi Phi^T (syrk, potrf)
// and forms no p x p matrix; the other branch factorises A.
// CHECK-LABEL: func.func @blr
// CHECK: arith.cmpi ult
// CHECK: scf.if
// CHECK: ulm_lamk.syrk
// CHECK: ulm_lamk.potrf
// CHECK: } else {
// CHECK: ulm_lamk.syrk
// CHECK: ulm_lamk.potrf
// CHECK: return

// Static sizes, N = 100 < p = 2000: only the Woodbury branch is left, with
// one Cholesky factorisation of a 100 x 100 matrix.
// CHECK-LABEL: func.func @blr_static
// CHECK-NOT: scf.if
// CHECK: ulm_lamk.potrf{{.*}}tensor<100x100xf64>
// CHECK-NOT: tensor<2000x2000xf64>
// CHECK: return

func.func @blr(%s.N: index, %s.p: index, %s.M: index, %p.Phi: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 1]}, %p.y: tensor<?x1xf64> {ulm_lam.in, ulm_lam.dims = [0]}, %p.Phis: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [2, 1]}, %p.alpha: f64, %p.beta: f64) -> (tensor<?x1xf64> {ulm_lam.out, ulm_lam.dims = [2]}, tensor<?x?xf64> {ulm_lam.out, ulm_lam.dims = [2, 2]}) {
  %v1 = ulm_lam.identity(%s.p, %s.p) : tensor<?x?xf64, #ulm_lam.props<identity>>
  %v2 = ulm_lam.scale %p.alpha, %v1 : f64, tensor<?x?xf64, #ulm_lam.props<identity>> -> tensor<?x?xf64>
  %v3 = ulm_lam.transpose %p.Phi : tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.mul %v3, %p.Phi : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v5 = ulm_lam.scale %p.beta, %v4 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v6 = ulm_lam.add %v2, %v5 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v7 = ulm_lam.assume %v6 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v8 = ulm_lam.scale %p.beta, %p.Phis : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v9 = ulm_lam.inverse %v7 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v10 = ulm_lam.mul %v8, %v9 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v11 = ulm_lam.mul %v10, %v3 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v12 = ulm_lam.mul %v11, %p.y : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
  %v13 = ulm_lam.mul %p.Phis, %v9 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v14 = ulm_lam.transpose %p.Phis : tensor<?x?xf64> -> tensor<?x?xf64>
  %v15 = ulm_lam.mul %v13, %v14 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %v12, %v15 : tensor<?x1xf64>, tensor<?x?xf64>
}

func.func @blr_static(%p.Phi: tensor<100x2000xf64> {ulm_lam.in}, %p.y: tensor<100x1xf64> {ulm_lam.in}, %p.Phis: tensor<50x2000xf64> {ulm_lam.in}, %p.alpha: f64, %p.beta: f64) -> (tensor<50x1xf64> {ulm_lam.out}, tensor<50x50xf64> {ulm_lam.out}) {
  %v1 = ulm_lam.identity : tensor<2000x2000xf64, #ulm_lam.props<identity>>
  %v2 = ulm_lam.scale %p.alpha, %v1 : f64, tensor<2000x2000xf64, #ulm_lam.props<identity>> -> tensor<2000x2000xf64>
  %v3 = ulm_lam.transpose %p.Phi : tensor<100x2000xf64> -> tensor<2000x100xf64>
  %v4 = ulm_lam.mul %v3, %p.Phi : tensor<2000x100xf64>, tensor<100x2000xf64> -> tensor<2000x2000xf64>
  %v5 = ulm_lam.scale %p.beta, %v4 : f64, tensor<2000x2000xf64> -> tensor<2000x2000xf64>
  %v6 = ulm_lam.add %v2, %v5 : tensor<2000x2000xf64>, tensor<2000x2000xf64> -> tensor<2000x2000xf64>
  %v7 = ulm_lam.assume %v6 : tensor<2000x2000xf64> -> tensor<2000x2000xf64, #ulm_lam.props<spd>>
  %v8 = ulm_lam.scale %p.beta, %p.Phis : f64, tensor<50x2000xf64> -> tensor<50x2000xf64>
  %v9 = ulm_lam.inverse %v7 : tensor<2000x2000xf64, #ulm_lam.props<spd>> -> tensor<2000x2000xf64>
  %v10 = ulm_lam.mul %v8, %v9 : tensor<50x2000xf64>, tensor<2000x2000xf64> -> tensor<50x2000xf64>
  %v11 = ulm_lam.mul %v10, %v3 : tensor<50x2000xf64>, tensor<2000x100xf64> -> tensor<50x100xf64>
  %v12 = ulm_lam.mul %v11, %p.y : tensor<50x100xf64>, tensor<100x1xf64> -> tensor<50x1xf64>
  %v13 = ulm_lam.mul %p.Phis, %v9 : tensor<50x2000xf64>, tensor<2000x2000xf64> -> tensor<50x2000xf64>
  %v14 = ulm_lam.transpose %p.Phis : tensor<50x2000xf64> -> tensor<2000x50xf64>
  %v15 = ulm_lam.mul %v13, %v14 : tensor<50x2000xf64>, tensor<2000x50xf64> -> tensor<50x50xf64>
  return %v12, %v15 : tensor<50x1xf64>, tensor<50x50xf64>
}
