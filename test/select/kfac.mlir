// RUN: ulm-lam-opt %s --pass-pipeline='builtin.module(func.func(ulm-lam-normalize, ulm-lam-infer-properties, ulm-lam-select-kernels), ulm-lam-strip-properties, canonicalize)' | FileCheck %s

// K-FAC (python/examples/kfac.py) with the symmetric matrices stored as
// their lower triangle: nothing is mirrored (no sycomplete).

// The running averages A <- rho A + c a^T a: scal, then one syrk with the
// run-time alpha c.
// CHECK-LABEL: func.func @kfac_factors
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: ulm_lamk.gescal
// CHECK-NEXT: ulm_lamk.syrk
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: return

// The damped inverses: adddiag on the stored triangle, potrf, potri.
// CHECK-LABEL: func.func @kfac_refresh
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: ulm_lamk.adddiag
// CHECK: ulm_lamk.potrf
// CHECK: ulm_lamk.potri
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: return

// The step reads the inverses with symm.
// CHECK-LABEL: func.func @kfac_precondition
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: ulm_lamk.symm
// CHECK-NOT: ulm_lamk.sycomplete
// CHECK: return

func.func @kfac_factors(%s.i: index, %s.o: index, %s.B: index, %p.A: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.inout, ulm_lam.dims = [0, 0]}, %p.G: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.inout, ulm_lam.dims = [1, 1]}, %p.a: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [2, 0]}, %p.g: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [2, 1]}, %p.rho: f64, %p.c: f64) -> (tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>> {ulm_lam.inout = 3 : i64, ulm_lam.dims = [0, 0]}, tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>> {ulm_lam.inout = 4 : i64, ulm_lam.dims = [1, 1]}) {
  %v1 = ulm_lam.scale %p.rho, %p.A : f64, tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64>
  %v2 = ulm_lam.transpose %p.a : tensor<?x?xf64> -> tensor<?x?xf64>
  %v3 = ulm_lam.mul %v2, %p.a : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.scale %p.c, %v3 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v5 = ulm_lam.add %v1, %v4 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v6 = ulm_lam.assume %v5 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>
  %v7 = ulm_lam.scale %p.rho, %p.G : f64, tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64>
  %v8 = ulm_lam.transpose %p.g : tensor<?x?xf64> -> tensor<?x?xf64>
  %v9 = ulm_lam.mul %v8, %p.g : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v10 = ulm_lam.scale %p.c, %v9 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v11 = ulm_lam.add %v7, %v10 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v12 = ulm_lam.assume %v11 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>
  %v13 = ulm_lam.assume %v6 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
  %v14 = ulm_lam.assume %v12 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
  return %v13, %v14 : tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>, tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
}

func.func @kfac_refresh(%s.i: index, %s.o: index, %p.A: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.G: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.in, ulm_lam.dims = [1, 1]}, %p.Ai: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.inout, ulm_lam.dims = [0, 0]}, %p.Gi: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.inout, ulm_lam.dims = [1, 1]}, %p.gamma: f64) -> (tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>> {ulm_lam.inout = 4 : i64, ulm_lam.dims = [0, 0]}, tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>> {ulm_lam.inout = 5 : i64, ulm_lam.dims = [1, 1]}) {
  %v1 = ulm_lam.identity(%s.i, %s.i) : tensor<?x?xf64, #ulm_lam.props<identity>>
  %v2 = ulm_lam.scale %p.gamma, %v1 : f64, tensor<?x?xf64, #ulm_lam.props<identity>> -> tensor<?x?xf64>
  %v3 = ulm_lam.add %p.A, %v2 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.assume %v3 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v5 = ulm_lam.inverse %v4 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v6 = ulm_lam.assume %v5 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>
  %v7 = ulm_lam.identity(%s.o, %s.o) : tensor<?x?xf64, #ulm_lam.props<identity>>
  %v8 = ulm_lam.scale %p.gamma, %v7 : f64, tensor<?x?xf64, #ulm_lam.props<identity>> -> tensor<?x?xf64>
  %v9 = ulm_lam.add %p.G, %v8 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v10 = ulm_lam.assume %v9 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<spd>>
  %v11 = ulm_lam.inverse %v10 : tensor<?x?xf64, #ulm_lam.props<spd>> -> tensor<?x?xf64>
  %v12 = ulm_lam.assume %v11 : tensor<?x?xf64> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>
  %v13 = ulm_lam.assume %v6 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
  %v14 = ulm_lam.assume %v12 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
  return %v13, %v14 : tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>, tensor<?x?xf64, #ulm_lam.props<lower_stored, symmetric>>
}

func.func @kfac_precondition(%s.o: index, %s.i: index, %s.B: index, %p.W: tensor<?x?xf64> {ulm_lam.inout, ulm_lam.dims = [0, 1]}, %p.Ai: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.in, ulm_lam.dims = [1, 1]}, %p.Gi: tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %p.a: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [2, 1]}, %p.g: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [2, 0]}, %p.lr: f64) -> (tensor<?x?xf64> {ulm_lam.inout = 3 : i64, ulm_lam.dims = [0, 1]}) {
  %v1 = ulm_lam.transpose %p.g : tensor<?x?xf64> -> tensor<?x?xf64>
  %v2 = ulm_lam.mul %p.Gi, %v1 : tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v3 = ulm_lam.mul %v2, %p.a : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  %v4 = ulm_lam.mul %v3, %p.Ai : tensor<?x?xf64>, tensor<?x?xf64, #ulm_lam.props<lower_stored, spd>> -> tensor<?x?xf64>
  %v5 = ulm_lam.scale %p.lr, %v4 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v6 = arith.constant -1.00000000000000000e+00 : f64
  %v7 = ulm_lam.scale %v6, %v5 : f64, tensor<?x?xf64> -> tensor<?x?xf64>
  %v8 = ulm_lam.add %p.W, %v7 : tensor<?x?xf64>, tensor<?x?xf64> -> tensor<?x?xf64>
  return %v8 : tensor<?x?xf64>
}
